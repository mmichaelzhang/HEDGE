"""Read-only packed contexts, identity indices and differentiable query reductions.

Prospective refinement only. No model fitting is started by this module. The
original preparation sources and source banks are never modified.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import torch
from torch.nn import functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'experiments'))
sys.path.insert(0, str(HERE.parent))
from paper_core import Batch
from paper_fast import TailIndex
from stable_ensemble_score import _group_logsum, SCORE_TYPE
from support_query_contract import PairPlan, query_rng


def sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(value)
    return h.hexdigest()


class PackedContext:
    """All arrays are CPU read-only memmaps; connection remapping is gather-only."""
    def __init__(self, directory, nd, nt, m, prefix='', remap=None):
        self.directory = pathlib.Path(directory)
        self.nd, self.nt, self.m = int(nd), int(nt), int(m)
        self.arrays = {key: np.load(self.directory / f'{prefix}{key}.npy', mmap_mode='r')
                       for key in ('nodes', 'connections', 'distances', 'lengths')}
        self.remap = None if remap is None else np.asarray(remap, dtype=np.int64)
        self.n, self.length = self.arrays['nodes'].shape
        if self.n != self.nd * self.m:
            raise ValueError('Context must have the frozen number of views for all heads')
        shapes = {'connections': (self.n, self.length - 1), 'distances': (self.n, self.length),
                  'lengths': (self.n,)}
        dtypes = {'nodes': np.int16, 'connections': np.uint16,
                  'distances': np.uint8, 'lengths': np.uint8}
        for key, arr in self.arrays.items():
            if arr.dtype != dtypes[key] or (key in shapes and arr.shape != shapes[key]):
                raise ValueError('Packed context schema mismatch: ' + key)
        if self.remap is not None and (self.remap.ndim != 1 or self.remap[0] != 0):
            raise ValueError('Invalid source-to-common tuple remapping')

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()

    def gather(self, ids, device='cpu'):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= self.n:
            raise ValueError('Invalid selected real-view IDs')
        nodes = np.asarray(self.arrays['nodes'][ids], dtype=np.int64)
        distances = np.asarray(self.arrays['distances'][ids], dtype=np.int64)
        connections = np.asarray(self.arrays['connections'][ids], dtype=np.int64)
        lengths = np.asarray(self.arrays['lengths'][ids], dtype=np.int64)
        if np.any(lengths < 1) or np.any(lengths > self.length):
            raise ValueError('Invalid retained-node length')
        # All prepared source views have head-major ordering, checked at index construction.
        if np.any(nodes[:, 0] != ids // self.m):
            raise ValueError('Selected view head does not match frozen head-major layout')
        if self.remap is not None:
            if connections.max(initial=0) >= len(self.remap):
                raise ValueError('Unknown source tuple ID')
            connections = self.remap[connections]
        valid = np.arange(self.length)[None, :] < lengths[:, None]
        return Batch(*(torch.as_tensor(a, device=device) for a in
                       (nodes, connections, distances, valid)))


def _csr_fill(keys, values, offsets, cursor, output):
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    unique, first, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    within = np.arange(len(keys)) - np.repeat(first, counts)
    positions = offsets[sorted_keys] + cursor[sorted_keys] + within
    if len(positions) and np.any(positions >= offsets[sorted_keys + 1]):
        raise ValueError('Unique-view recount exceeds declared CSR count')
    output[positions] = values[order]
    cursor[unique] += counts


def build_identity_index(context, counts, directory, lineage, chunk=2048):
    """Independent unique(view,target) scan; no labels or model outputs accepted.

    CSR includes each view once per target. Source pair counts determine capacity;
    an independent scan must exactly fill both CSR arrays or construction fails.
    """
    directory = pathlib.Path(directory)
    if directory.exists():
        raise FileExistsError('Refuse to overwrite identity-index evidence')
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (context.nd, context.nt) or np.any(counts < 0):
        raise ValueError('Invalid unique-view counts')
    if context.n > np.iinfo(np.int32).max:
        raise ValueError('View IDs exceed declared int32 index storage')
    directory.mkdir(parents=True)
    pair_offsets = np.r_[0, counts.ravel().cumsum()]
    target_offsets = np.r_[0, counts.sum(0).cumsum()]
    total = int(pair_offsets[-1])
    if total != int(target_offsets[-1]):
        raise ValueError('Pair and target unique-view totals differ')
    np.save(directory / 'pair_offsets.npy', pair_offsets)
    np.save(directory / 'target_offsets.npy', target_offsets)
    pv = np.lib.format.open_memmap(directory / 'pair_views.npy', mode='w+', dtype=np.int32, shape=(total,))
    tv = np.lib.format.open_memmap(directory / 'target_views.npy', mode='w+', dtype=np.int32, shape=(total,))
    pc, tc = np.zeros(context.nd * context.nt, np.int64), np.zeros(context.nt, np.int64)
    heads_seen = np.zeros(context.nd, np.int64)
    for start in range(0, context.n, chunk):
        row = np.asarray(context.arrays['nodes'][start:start + chunk])
        size = np.asarray(context.arrays['lengths'][start:start + chunk])
        head = row[:, 0].astype(np.int64)
        if np.any(head != np.arange(start, start + len(row)) // context.m):
            raise ValueError('Source is not the frozen all-head head-major bank')
        np.add.at(heads_seen, head, 1)
        rr, cc = np.nonzero(np.arange(context.length - 1)[None, :] < (size - 1)[:, None])
        target = row[rr, cc + 1].astype(np.int64) - context.nd
        if np.any(target < 0) or np.any(target >= context.nt):
            raise ValueError('Retained tail is not a target')
        unique = np.unique(rr.astype(np.int64) * context.nt + target)
        rr, target = unique // context.nt, unique % context.nt
        view = (start + rr).astype(np.int32)
        pair = head[rr] * context.nt + target
        _csr_fill(pair, view, pair_offsets, pc, pv)
        _csr_fill(target, view, target_offsets, tc, tv)
    if (not np.array_equal(pc, counts.ravel()) or not np.array_equal(tc, counts.sum(0))
            or not np.all(heads_seen == context.m)):
        raise ValueError('Independent view recount did not match declared coverage')
    pv.flush(); tv.flush()
    del pv, tv
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(directory.glob('*.npy'))}
    record = {'scope': 'identity-only unique-view inverted index; no labels or model scores',
        'nD': context.nd, 'nT': context.nt, 'views': context.n, 'views_per_head': context.m,
        'total_unique_view_target_memberships': total, 'lineage': lineage, 'files': files}
    (directory / 'COMPLETE.json').write_text(json.dumps(record, indent=2))
    return record


class IdentityIndex:
    def __init__(self, directory, expected_lineage=None):
        self.directory = pathlib.Path(directory)
        self.record = json.loads((self.directory / 'COMPLETE.json').read_text())
        if expected_lineage is not None and self.record['lineage'] != expected_lineage:
            raise ValueError('Identity index lineage differs from the selected context')
        names = {'pair_offsets.npy', 'target_offsets.npy', 'pair_views.npy', 'target_views.npy'}
        if set(self.record['files']) != names:
            raise ValueError('Unexpected CSR index file inventory')
        for name, rec in self.record['files'].items():
            path = self.directory / name
            if path.stat().st_size != rec['bytes'] or sha256(path) != rec['sha256']:
                raise ValueError('Identity index hash mismatch: ' + name)
        for key in ('pair_offsets', 'target_offsets', 'pair_views', 'target_views'):
            setattr(self, key, np.load(self.directory / f'{key}.npy', mmap_mode='r'))
        self.nd, self.nt, self.m = (int(self.record[k]) for k in ('nD', 'nT', 'views_per_head'))
        if (self.pair_offsets.dtype != np.int64 or self.target_offsets.dtype != np.int64
                or self.pair_views.dtype != np.int32 or self.target_views.dtype != np.int32):
            raise ValueError('CSR storage dtype differs from the frozen identity-index contract')
        if (len(self.pair_offsets) != self.nd * self.nt + 1
                or len(self.target_offsets) != self.nt + 1
                or self.pair_offsets[0] != 0 or self.target_offsets[0] != 0
                or np.any(np.diff(self.pair_offsets) < 0) or np.any(np.diff(self.target_offsets) < 0)
                or self.pair_offsets[-1] != len(self.pair_views)
                or self.target_offsets[-1] != len(self.target_views)):
            raise ValueError('Invalid CSR offsets or membership capacity')

    def close(self):
        for key in ('pair_offsets', 'target_offsets', 'pair_views', 'target_views'):
            getattr(self, key)._mmap.close()

    def plan(self, head, target, rng):
        if not 0 <= head < self.nd or not self.nd <= target < self.nd + self.nt:
            raise ValueError('Query identity outside bank universe')
        pair = head * self.nt + target - self.nd
        lo, hi = self.pair_offsets[pair:pair + 2]
        if hi > lo:
            view = int(self.pair_views[int(rng.integers(lo, hi))])
            return PairPlan(head, target, 'cooccurrence', pair_view=view)
        head_views = tuple(map(int, rng.integers(head * self.m, (head + 1) * self.m, size=2)))
        lo, hi = self.target_offsets[target - self.nd:target - self.nd + 2]
        if hi > lo:
            target_views = tuple(map(int, self.target_views[rng.integers(lo, hi, size=2)]))
            return PairPlan(head, target, 'global_context', head_views=head_views, target_views=target_views)
        return PairPlan(head, target, 'input_target', head_views=head_views)


def plans_for_queries(index, pairs, fold, update):
    return [index.plan(int(head), int(target) + index.nd,
                      query_rng(fold, update, slot, int(head), int(target) + index.nd))
            for slot, (head, target) in enumerate(pairs)]


def plan_view_ids(plans):
    ids = []
    for plan in plans:
        ids.extend([plan.pair_view] if plan.branch == 'cooccurrence'
                   else list(plan.head_views) + list(plan.target_views))
    return np.unique(np.asarray(ids, dtype=np.int64))


def reduce_encoded_plans(model, encoded, plans, view_ids):
    """One encoded state per UNIQUE view, but every repeated draw keeps its weight."""
    local = {int(view): row for row, view in enumerate(view_ids)}
    head_rows, target_rows, weights, absent = [], [], [], []
    for plan in plans:
        if plan.branch == 'cooccurrence':
            head_rows.append([local[plan.pair_view]] * 2)
            target_rows.append([local[plan.pair_view]] * 2)
            weights.append([1., 0.]); absent.append(False)
        elif plan.branch in ('global_context', 'input_target'):
            if len(plan.head_views) != 2:
                raise ValueError('Expected frozen two-view head mean')
            head_rows.append([local[v] for v in plan.head_views])
            if plan.branch == 'global_context':
                if len(plan.target_views) != 2:
                    raise ValueError('Expected frozen two-view target mean')
                target_rows.append([local[v] for v in plan.target_views]); absent.append(False)
            else:
                target_rows.append([0, 0]); absent.append(True)
            weights.append([.5, .5])
        else:
            raise ValueError('Unknown planned supervision branch')
    device = encoded.nodes.device
    hr = torch.as_tensor(head_rows, dtype=torch.long, device=device)
    tr = torch.as_tensor(target_rows, dtype=torch.long, device=device)
    w = encoded.nodes.new_tensor(weights)
    heads = torch.as_tensor([p.head for p in plans], device=device)
    targets = torch.as_tensor([p.target for p in plans], device=device)
    absent = torch.as_tensor(absent, dtype=torch.bool, device=device)
    if torch.any(encoded.batch.nodes[hr, 0] != heads[:, None]):
        raise ValueError('A planned source view has an incorrect head')
    hm = (encoded.nodes[hr, 0] * w[:, :, None]).sum(1)
    match = (encoded.batch.nodes[tr] == targets[:, None, None]) & encoded.batch.valid[tr]
    match[:, :, 0] = False
    occurrences = match.sum(-1)
    if torch.any((occurrences == 0) & (w > 0) & ~absent[:, None]):
        raise ValueError('A planned target view lacks the requested target')
    per_view = (encoded.nodes[tr] * match[:, :, :, None]).sum(2) / occurrences.clamp_min(1)[:, :, None]
    tm = (per_view * w[:, :, None]).sum(1)
    if absent.any():
        tm = tm.clone()
        tm[absent] = model.node_input(targets[absent])
    return model.pair_logit(hm, tm)


def encode_query_plans(model, context, plans, device):
    view_ids = plan_view_ids(plans)
    batch = context.gather(view_ids, device)
    encoded = model.encode(batch)  # exactly one call: one dropout realization per distinct view
    return reduce_encoded_plans(model, encoded, plans, view_ids), view_ids


@torch.no_grad()
def streaming_score(model, context, bs=512, head_chunk=16, target_chunk=512):
    """Same stable full-view computation as stable_ensemble_score.score, streamed.

    This function reads node identities, model states and the view bank only. It
    accepts no labels and makes no test/validation selections.
    """
    model.eval()
    device = next(model.parameters()).device
    nd, nt = context.nd, context.nt
    hs = torch.zeros(nd, model.d, device=device)
    hc = torch.zeros(nd, device=device)
    ts = torch.zeros(nt, model.d, device=device)
    tc = torch.zeros(nt, device=device)
    log_p = torch.full((nd * nt,), -torch.inf, dtype=torch.float64, device=device)
    log_q = torch.full_like(log_p, -torch.inf)
    counts = torch.zeros(nd * nt, dtype=torch.long, device=device)
    for start in range(0, context.n, bs):
        batch = context.gather(np.arange(start, min(start + bs, context.n)), device)
        encoded = model.encode(batch)
        heads = batch.nodes[:, 0]
        hs.index_add_(0, heads, encoded.nodes[:, 0])
        hc.index_add_(0, heads, torch.ones(len(heads), device=device))
        index = TailIndex.from_batch(batch)
        if not len(index.row):
            continue
        means = index.mean_states(encoded)
        targets = index.target - nd
        if (targets < 0).any() or (targets >= nt).any():
            raise ValueError('Retained tail outside target universe')
        logits = model.pair_logit(encoded.nodes[index.row, 0], means).double()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite pair logit')
        pair = heads[index.row] * nt + targets
        unique, inverse = torch.unique(pair, sorted=True, return_inverse=True)
        log_p[unique] = torch.logaddexp(log_p[unique], _group_logsum(F.logsigmoid(logits), inverse, len(unique)))
        log_q[unique] = torch.logaddexp(log_q[unique], _group_logsum(F.logsigmoid(-logits), inverse, len(unique)))
        counts.index_add_(0, pair, torch.ones_like(pair))
        ts.index_add_(0, targets, means)
        tc.index_add_(0, targets, torch.ones(len(targets), device=device))
    if (hc == 0).any():
        raise ValueError('Every drug requires a full evaluation context')
    hm, tm = hs / hc[:, None], ts / tc.clamp_min(1)[:, None]
    missing = tc == 0
    if missing.any():
        tm[missing] = model.node_input(torch.arange(nd, nd + nt, device=device)[missing])
    counts = counts.view(nd, nt)
    log_p, log_q = log_p.view(nd, nt), log_q.view(nd, nt)
    result = torch.empty(nd, nt, dtype=torch.float64, device=device)
    for i in range(0, nd, head_chunk):
        for j in range(0, nt, target_chunk):
            fallback = model.pair_logit(hm[i:i + head_chunk, None], tm[None, j:j + target_chunk]).double()
            c = counts[i:i + head_chunk, j:j + target_chunk]
            observed = log_p[i:i + head_chunk, j:j + target_chunk] - log_q[i:i + head_chunk, j:j + target_chunk]
            result[i:i + head_chunk, j:j + target_chunk] = torch.where(c > 0, observed, fallback)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite stable ensemble/fallback score')
    coverage = {'global_absent_targets': int(missing.sum()),
        'cooccurrence_pair_fraction': float((counts > 0).float().mean()),
        'input_only_fallback_pair_fraction': float(((counts == 0) & missing[None]).float().mean())}
    return result.cpu().numpy(), counts.cpu().numpy(), coverage
