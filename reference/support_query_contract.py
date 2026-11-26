"""CPU contracts for a prospective support/query refinement; never trains.

Only explicitly supplied optimization-training labels enter partitioning. View
selection accepts node identities, not labels. No frozen source is modified.
"""
from dataclasses import dataclass
import hashlib

import numpy as np


MASTER_SEED = 20260930
STATIC_DIMENSION = 1652
DTI_SLICES = ((612, 740), (1524, 1652))


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def checked_pairs(pairs, n_drugs, n_targets):
    value = np.asarray(pairs)
    if value.ndim != 2 or value.shape[1] != 2 or value.dtype.kind not in 'iu':
        raise ValueError('Pairs must be an integer N by 2 array')
    value = value.astype(np.int64, copy=False)
    if len(value) and (np.any(value < 0) or np.any(value[:, 0] >= n_drugs)
                       or np.any(value[:, 1] >= n_targets)):
        raise ValueError('Pair identity outside entity universe')
    if len(np.unique(value, axis=0)) != len(value):
        raise ValueError('Duplicate optimization-training pair')
    return value


def partition_training_pairs(train_pos, train_neg, n_drugs, n_targets,
                             fold=0, blocks=5, master_seed=MASTER_SEED):
    """Per-drug shuffled round robin, independent positive/negative RNG streams.

    Return blocks in lexicographic identity order, without input-order dependence.
    Randomized offsets allow a one-pair drug to enter any of the five blocks.
    Record explicit training-label heads; context banks use ALL known drug heads,
    including drugs without a supervised label in the current query block.
    """
    if min(n_drugs, n_targets) < 1 or blocks < 2 or fold < 0:
        raise ValueError('Invalid partition dimensions')
    pp = checked_pairs(train_pos, n_drugs, n_targets)
    nn = checked_pairs(train_neg, n_drugs, n_targets)
    pset, nset = set(map(tuple, pp)), set(map(tuple, nn))
    if pset & nset:
        raise ValueError('Contradictory optimization-training labels')
    if not len(pp) or not len(nn):
        raise ValueError('Both optimization-training labels are required')
    result = [{'positive': [], 'negative': []} for _ in range(blocks)]
    for role, pairs in enumerate((pp, nn)):
        key = ('positive', 'negative')[role]
        for head in np.unique(pairs[:, 0]):
            targets = np.sort(pairs[pairs[:, 0] == head, 1])
            rng = np.random.default_rng(np.random.SeedSequence(
                [master_seed, fold, role, int(head)]))
            targets = rng.permutation(targets)
            offset = int(rng.integers(blocks))
            for j, target in enumerate(targets):
                result[(offset + j) % blocks][key].append((int(head), int(target)))
    for block in result:
        for key in ('positive', 'negative'):
            block[key] = np.asarray(sorted(block[key]), dtype=np.int64).reshape(-1, 2)
        block['pairs'] = np.concatenate((block['positive'], block['negative']))
        block['labels'] = np.r_[np.ones(len(block['positive']), np.float32),
                                 np.zeros(len(block['negative']), np.float32)]
    heads = np.unique(np.r_[pp[:, 0], nn[:, 0]])
    return {'blocks': result, 'train_heads': heads,
            'context_heads': np.arange(n_drugs, dtype=np.int64),
            'positive_weight': min(200., len(nn) / len(pp)),
            'seed': master_seed, 'fold': fold}


def query_rng(fold, step, slot, head, target, master_seed=MASTER_SEED):
    """Arm-independent per-query-occurrence stream; branch lengths cannot drift.

    `slot` distinguishes duplicate occurrences within one batch. Neither label,
    block graph, arm identity nor earlier random draws enters the seed.
    """
    if min(fold, step, slot, head, target, master_seed) < 0:
        raise ValueError('Query RNG keys must be nonnegative')
    return np.random.default_rng(np.random.SeedSequence(
        [master_seed, 17, fold, step, slot, head, target]))


def static_only_features(features):
    """Preserve frozen Luo static coordinates; zero both train-DTI SVD blocks."""
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != STATIC_DIMENSION or not np.isfinite(x).all():
        raise ValueError('Expected finite frozen Luo 1652-column feature matrix')
    out = x.copy()
    for lo, hi in DTI_SLICES:
        out[:, lo:hi] = 0
    return out


def pair_mask(pairs, n_drugs, n_targets):
    pairs = checked_pairs(pairs, n_drugs, n_targets)
    mask = np.zeros((n_drugs, n_targets), dtype=np.bool_)
    mask[pairs[:, 0], pairs[:, 1]] = True
    return mask


def filter_dti_csr(offset, neighbor, relation, support_mask):
    """Filter relation-0 edges in BOTH directions, preserving all auxiliary order.

    The full input graph must contain both directions of every support edge.
    Reject malformed relation-0 endpoint types instead of treating them as aux.
    Distances/walks are not accepted or returned: they must be rebuilt afterward.
    """
    off = np.asarray(offset, dtype=np.int64)
    nb = np.asarray(neighbor, dtype=np.int64)
    rel = np.asarray(relation, dtype=np.int64)
    support = np.asarray(support_mask, dtype=np.bool_)
    if support.ndim != 2 or off.ndim != 1 or len(off) < 2:
        raise ValueError('Invalid CSR or support mask')
    nd, nt = support.shape
    total = int(off[-1])
    if (off[0] != 0 or np.any(np.diff(off) < 0) or len(nb) < total
            or len(rel) < total or len(off) - 1 < nd + nt):
        raise ValueError('Invalid CSR arrays')
    nb, rel = nb[:total], rel[:total]
    src = np.repeat(np.arange(len(off) - 1), np.diff(off))
    if np.any(nb < 0) or np.any(nb >= len(off) - 1) or np.any(rel < 0):
        raise ValueError('Invalid edge endpoint/relation')
    dti = rel == 0
    forward = (src < nd) & (nb >= nd) & (nb < nd + nt)
    reverse = (nb < nd) & (src >= nd) & (src < nd + nt)
    if np.any(dti & ~(forward | reverse)):
        raise ValueError('Relation 0 must connect a drug and target')
    keep = ~dti
    dti_index = np.flatnonzero(dti)
    drug = np.where(forward[dti], src[dti], nb[dti])
    target = np.where(forward[dti], nb[dti], src[dti]) - nd
    keep[dti_index] = support[drug, target]
    for orient in (forward, reverse):
        selected = dti & orient & keep
        observed = np.zeros_like(support)
        observed_drug = np.where(forward[selected], src[selected], nb[selected])
        observed_target = np.where(forward[selected], nb[selected], src[selected]) - nd
        observed[observed_drug, observed_target] = True
        if not np.array_equal(observed, support):
            raise ValueError('Support edges lack a complete forward/reverse graph image')
    counts = np.bincount(src[keep], minlength=len(off) - 1)
    return np.r_[0, counts.cumsum()].astype(np.int64), nb[keep], rel[keep]


@dataclass(frozen=True)
class PairPlan:
    head: int
    target: int                       # global target node id
    branch: str
    pair_view: int | None = None
    head_views: tuple = ()
    target_views: tuple = ()


class RealViewIndex:
    """Reference CPU identity index for small contracts, not a large-bank format.

    Duplicate occurrences of one target count as ONE view in every distribution.
    Encoded target states are averaged over duplicates within the sampled view.
    """
    def __init__(self, nodes, n_drugs, n_targets):
        self.nodes = tuple(tuple(map(int, value)) for value in nodes)
        self.n_drugs, self.n_targets = int(n_drugs), int(n_targets)
        self.by_head, self.by_target, self.by_pair = {}, {}, {}
        for i, row in enumerate(self.nodes):
            if not row or not 0 <= row[0] < n_drugs:
                raise ValueError('Real view requires one valid drug head')
            if any(not n_drugs <= t < n_drugs + n_targets for t in row[1:]):
                raise ValueError('Only target nodes are retained after the head')
            self.by_head.setdefault(row[0], []).append(i)
            for target in sorted(set(row[1:])):
                self.by_target.setdefault(target, []).append(i)
                self.by_pair.setdefault((row[0], target), []).append(i)

    def plan(self, head, target, rng, head_draws=2, target_draws=2):
        """Target is a global node id; labels cannot influence this API."""
        if (head_draws != 2 or target_draws != 2 or head not in self.by_head
                or not self.n_drugs <= target < self.n_drugs + self.n_targets):
            raise ValueError('Unknown query identity or non-frozen MC budget')
        together = self.by_pair.get((head, target), ())
        if together:
            return PairPlan(head, target, 'cooccurrence', int(rng.choice(together)))
        # Separate uniform draws are independent conditional on this RNG stream.
        hv = tuple(map(int, rng.choice(self.by_head[head], 2, replace=True)))
        available = self.by_target.get(target, ())
        if available:
            tv = tuple(map(int, rng.choice(available, 2, replace=True)))
            return PairPlan(head, target, 'global_context', head_views=hv, target_views=tv)
        return PairPlan(head, target, 'input_target', head_views=hv)


def differentiable_pair_logit(plan, nodes, encoded_states, pair_logit, node_input):
    """Tiny reference reduction with no detach, no fitting, and no probability mean.

    `encoded_states[view]` contains differentiable states at all valid node slots.
    Duplicate draws retain multiplicity. A production runner may batch/deduplicate
    forwards only if it preserves this reduction and its dropout contract.
    """
    import torch

    def target_state(view):
        positions = [j for j, node in enumerate(nodes[view])
                     if j and node == plan.target]
        if not positions:
            raise ValueError('Planned real view does not contain its target')
        return encoded_states[view][positions].mean(0)

    if plan.branch == 'cooccurrence':
        view = plan.pair_view
        if nodes[view][0] != plan.head:
            raise ValueError('Incorrect pair-view head')
        head, target = encoded_states[view][0], target_state(view)
    elif plan.branch in ('global_context', 'input_target'):
        if len(plan.head_views) != 2 or any(nodes[v][0] != plan.head for v in plan.head_views):
            raise ValueError('Invalid head-context plan')
        head = torch.stack([encoded_states[v][0] for v in plan.head_views]).mean(0)
        if plan.branch == 'global_context':
            if len(plan.target_views) != 2:
                raise ValueError('Invalid target-context plan')
            target = torch.stack([target_state(v) for v in plan.target_views]).mean(0)
        else:
            target = node_input(plan.target)
    else:
        raise ValueError('Unknown fallback branch')
    return pair_logit(head, target)
