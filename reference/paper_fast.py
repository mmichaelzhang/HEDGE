"""Vectorized/streaming equivalents of paper_core definitions for small HBNs.

Index plans may be prepared once for fixed sample banks. Streaming inference
keeps dense head-by-target sums/counts but not all contextual views. It is not
appropriate for a 258K-by-22K dense candidate matrix without further chunking.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class TailIndex:
    occurrence: torch.Tensor
    inverse: torch.Tensor
    occurrence_count: torch.Tensor
    row: torch.Tensor
    target: torch.Tensor
    positive: torch.Tensor
    row_cpu: np.ndarray
    target_cpu: np.ndarray
    heads_cpu: np.ndarray
    batch_size: int

    @classmethod
    def from_batch(cls, batch, train_positives=None, allowed_targets=None):
        """Build once per fixed collated bank/minibatch. Targets are GLOBAL IDs."""
        nodes = batch.nodes.detach().cpu().numpy()
        valid = batch.valid.detach().cpu().numpy().copy()
        valid[:, 0] = False
        rows, cols = np.where(valid)
        targets = nodes[rows, cols]
        if allowed_targets is not None:
            keep = np.fromiter((int(t) in allowed_targets for t in targets), bool)
            rows, cols, targets = rows[keep], cols[keep], targets[keep]
        pairs = np.stack((rows, targets), 1)
        groups, inverse = np.unique(pairs, axis=0, return_inverse=True)
        group_rows, group_targets = groups[:, 0], groups[:, 1]
        count = np.bincount(inverse, minlength=len(groups))
        positives = {} if train_positives is None else {
            int(u): set(map(int, ts)) for u, ts in train_positives.items()}
        labels = np.fromiter((int(t) in positives.get(int(nodes[r, 0]), set())
                              for r, t in groups), bool, count=len(groups))
        def tensor(a, dtype=torch.long):
            return torch.as_tensor(a, dtype=dtype, device=batch.nodes.device)
        return cls(tensor(rows * nodes.shape[1] + cols), tensor(inverse),
                   tensor(count), tensor(group_rows), tensor(group_targets),
                   tensor(labels, torch.bool), group_rows, group_targets,
                   nodes[:, 0].copy(), len(nodes))

    def mean_states(self, encoded):
        values = encoded.nodes.reshape(-1, encoded.nodes.shape[-1])[self.occurrence]
        result = values.new_zeros((len(self.row), encoded.nodes.shape[-1]))
        return result.index_add(0, self.inverse, values) / self.occurrence_count.clamp_min(1)[:, None]


def contrastive_loss(encoded, index, tau=0.2, normalize=False, reduction="mean"):
    """Same unique-tail, positive-free-sample and reduction rules as core."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be sum or mean")
    if not len(index.row):
        return encoded.nodes.sum() * 0
    tails = index.mean_states(encoded)
    anchors = encoded.nodes[index.row, 0]
    if normalize:
        tails, anchors = F.normalize(tails, dim=-1), F.normalize(anchors, dim=-1)
    logits = (tails * anchors).sum(-1) / tau
    maximum = logits.new_full((index.batch_size,), -torch.inf)
    maximum.scatter_reduce_(0, index.row, logits.detach(), reduce="amax", include_self=True)
    shifted = logits - maximum[index.row]
    denom = logits.new_zeros(index.batch_size).index_add(0, index.row, shifted.exp())
    log_prob = shifted - denom[index.row].log()
    npositive = logits.new_zeros(index.batch_size).index_add(
        0, index.row, index.positive.to(logits.dtype))
    values = -log_prob * index.positive / npositive[index.row].clamp_min(1)
    result = values.sum()
    return result if reduction == "sum" else result / index.batch_size


def decoder_bce(model, encoded, index, pos_weight=None):
    if not len(index.row):
        return encoded.nodes.sum() * 0
    logits = model.pair_logit(encoded.nodes[index.row, 0], index.mean_states(encoded))
    weight = None if pos_weight is None else logits.new_tensor(pos_weight)
    return F.binary_cross_entropy_with_logits(logits, index.positive.to(logits.dtype),
                                              pos_weight=weight)


class StreamingEnsemble:
    """Average probabilities over unique co-occurring views; no per-pair Python MLP.

    Requires all requested heads to have at least one sample. add() accepts only
    full TailIndex plans, not plans filtered to training candidates. Unrequested
    heads still contribute to global target means. All state tensors use one
    model/device/dtype; no training gradients are retained.
    """
    def __init__(self, model, heads, targets):
        self.model = model
        self.heads = tuple(map(int, heads))
        self.targets = tuple(map(int, targets))
        if not self.heads or not self.targets:
            raise ValueError("Nonempty head and target sets required")
        if len(set(self.heads)) != len(self.heads) or len(set(self.targets)) != len(self.targets):
            raise ValueError("Duplicate requested entity IDs")
        self.head_map = {u: i for i, u in enumerate(self.heads)}
        self.target_map = {t: i for i, t in enumerate(self.targets)}
        self.initialized = False

    def _initialize(self, encoded):
        self.hsum = encoded.nodes.new_zeros((len(self.heads), encoded.nodes.shape[-1]))
        self.tsum = encoded.nodes.new_zeros((len(self.targets), encoded.nodes.shape[-1]))
        self.hcount = self.hsum.new_zeros(len(self.heads))
        self.tcount = self.tsum.new_zeros(len(self.targets))
        self.psum = encoded.nodes.new_zeros(len(self.heads) * len(self.targets))
        self.count = torch.zeros_like(self.psum, dtype=torch.long)
        self.initialized = True

    @torch.no_grad()
    def add(self, encoded, index=None):
        if index is None:
            index = TailIndex.from_batch(encoded.batch)
        if not self.initialized:
            self._initialize(encoded)
        device = encoded.nodes.device
        head_rows = np.fromiter((self.head_map.get(int(u), -1) for u in index.heads_cpu), np.int64)
        head_keep = np.flatnonzero(head_rows >= 0)
        hi = torch.as_tensor(head_rows[head_keep], device=device)
        hk = torch.as_tensor(head_keep, device=device)
        self.hsum.index_add_(0, hi, encoded.nodes[hk, 0])
        self.hcount.index_add_(0, hi, self.hcount.new_ones(len(hi)))
        if not len(index.row):
            return
        states = index.mean_states(encoded)
        target_rows = np.fromiter((self.target_map.get(int(t), -1) for t in index.target_cpu), np.int64)
        target_keep = np.flatnonzero(target_rows >= 0)
        ti = torch.as_tensor(target_rows[target_keep], device=device)
        tk = torch.as_tensor(target_keep, device=device)
        self.tsum.index_add_(0, ti, states[tk])
        self.tcount.index_add_(0, ti, self.tcount.new_ones(len(ti)))
        # Score all observed requested pairs in a single MLP batch.
        pair_head_rows = head_rows[index.row_cpu]
        pair_keep = np.flatnonzero((pair_head_rows >= 0) & (target_rows >= 0))
        pk = torch.as_tensor(pair_keep, device=device)
        flat_pair = torch.as_tensor(pair_head_rows[pair_keep] * len(self.targets)
                                    + target_rows[pair_keep], device=device)
        probabilities = torch.sigmoid(self.model.pair_logit(
            encoded.nodes[index.row[pk], 0], states[pk]))
        self.psum.index_add_(0, flat_pair, probabilities)
        self.count.index_add_(0, flat_pair, torch.ones_like(flat_pair))

    @torch.no_grad()
    def finalize(self, fallback="error", fallback_target_states=None,
                 head_chunk=16, target_chunk=512):
        if not self.initialized:
            raise ValueError("No batches were accumulated")
        if fallback not in {"error", "global_target_mean"}:
            raise ValueError("Unknown fallback policy")
        if (self.hcount == 0).any():
            raise ValueError("At least one requested head has no sampled representation")
        counts = self.count.view(len(self.heads), len(self.targets))
        missing = counts == 0
        if fallback == "error" and missing.any():
            raise ValueError("At least one pair never co-occurred")
        result = (self.psum / self.count.clamp_min(1)).view_as(counts)
        kinds = torch.zeros_like(counts)
        if not missing.any():
            return result, counts, kinds
        h = self.hsum / self.hcount[:, None]
        t = self.tsum / self.tcount.clamp_min(1)[:, None]
        absent = self.tcount == 0
        # A supplied [n_targets,d] tensor follows self.targets ordering; a dict
        # uses global IDs, matching core. It is only read for absent targets.
        if absent.any():
            if fallback_target_states is None:
                raise ValueError("Some targets absent globally; explicit states required")
            if isinstance(fallback_target_states, torch.Tensor):
                if fallback_target_states.shape != t.shape:
                    raise ValueError("Fallback tensor shape mismatch")
                t[absent] = fallback_target_states.to(t)[absent]
            else:
                for i in torch.where(absent)[0].cpu().tolist():
                    target = self.targets[i]
                    if target not in fallback_target_states:
                        raise ValueError(f"Missing explicit fallback for target {target}")
                    t[i] = fallback_target_states[target].to(t)
        # Vectorized Hadamard MLP over bounded Cartesian blocks, then override
        # only missing entries. Observed scores are never blended with fallback.
        for a in range(0, len(self.heads), head_chunk):
            for b in range(0, len(self.targets), target_chunk):
                sel = missing[a:a + head_chunk, b:b + target_chunk]
                if not sel.any():
                    continue
                probability = torch.sigmoid(self.model.pair_logit(
                    h[a:a + head_chunk, None], t[None, b:b + target_chunk]))
                result[a:a + head_chunk, b:b + target_chunk] = torch.where(
                    sel, probability, result[a:a + head_chunk, b:b + target_chunk])
        kinds[missing] = 1
        kinds[missing & absent[None]] = 2
        return result, counts, kinds
