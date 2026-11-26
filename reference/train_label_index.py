"""Restrict supervised sampled candidates to explicit optimization-training labels.

This does not filter encoder context or the unsupervised observation term.
It never consults validation/test pair identities to build the loss mask.
"""
from types import SimpleNamespace
import numpy as np
import torch
from paper_fast import TailIndex


def declared_training_mask(train_pos, train_neg, n_drugs, n_targets):
    known = np.zeros((n_drugs, n_targets), dtype=bool)
    pos = np.zeros_like(known)
    for name, source in [('positive', train_pos), ('negative', train_neg)]:
        a = np.asarray(source, dtype=np.int64)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f'Invalid {name} training pair shape')
        if len(a) and ((a < 0).any() or (a[:, 0] >= n_drugs).any() or (a[:, 1] >= n_targets).any()):
            raise ValueError(f'{name} training pair outside node universe')
        if known[a[:, 0], a[:, 1]].any():
            raise ValueError('Pair assigned to both training label classes')
        if len(np.unique(a, axis=0)) != len(a):
            raise ValueError(f'Duplicate {name} training pair')
        known[a[:, 0], a[:, 1]] = True
        if name == 'positive':
            pos[a[:, 0], a[:, 1]] = True
    positives = {int(d): set((np.flatnonzero(pos[d]) + n_drugs).tolist())
                 for d in np.flatnonzero(pos.any(axis=1))}
    return known, positives


def training_tail_index(batch, train_positive_map, known_pairs, n_drugs):
    """Index explicit training-label candidates without changing `batch` itself."""
    nodes = batch.nodes.detach().cpu().numpy()
    valid = batch.valid.detach().cpu().numpy().copy()
    valid[:, 0] = False
    rows, cols = np.where(valid)
    heads = nodes[rows, 0]
    targets = nodes[rows, cols] - n_drugs
    if len(rows) and ((heads < 0).any() or (heads >= known_pairs.shape[0]).any()
                      or (targets < 0).any() or (targets >= known_pairs.shape[1]).any()):
        raise ValueError('Expected one drug head and target-only retained tails')
    valid[rows, cols] = np.asarray(known_pairs, dtype=bool)[heads, targets]
    proxy = SimpleNamespace(nodes=batch.nodes,
                            valid=torch.as_tensor(valid, dtype=torch.bool, device=batch.nodes.device))
    return TailIndex.from_batch(proxy, train_positive_map)
