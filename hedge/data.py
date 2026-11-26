"""Numeric NPZ input contract for the legacy demonstration.

Required: scalar nD/nT; binary train_pos_mask[nD,nT]; integer N-by-2 arrays
train_pos, test_pos, test_neg. Optional val_pos/val_neg/train_neg use the same
local (drug, target) IDs. The graph mask must equal the training positives.
This loader does not construct datasets, evaluate predictions, or establish
the provenance of upstream features. See reference/ for the later core.
"""
import os
import numpy as np

REQUIRED = ('nD', 'nT', 'train_pos_mask', 'train_pos', 'test_pos', 'test_neg')


def _dimension(value, name):
    value = np.asarray(value)
    if value.size != 1 or value.dtype.kind not in 'iu' or int(value.item()) < 1:
        raise ValueError(f'{name} must be a positive integer scalar')
    return int(value.item())


def _pairs(value, name, nd, nt):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != 2 or value.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be an integer N-by-2 array')
    if len(value) and ((value < 0).any() or (value[:, 0] >= nd).any()
                      or (value[:, 1] >= nt).any()):
        raise ValueError(f'{name} contains an out-of-range ID')
    if len(np.unique(value, axis=0)) != len(value):
        raise ValueError(f'{name} contains duplicate pairs')
    return value


def check_no_leakage(fold):
    """Reject malformed labels, overlapping splits, or nontraining graph edges."""
    missing = [key for key in REQUIRED if key not in fold]
    if missing:
        raise KeyError(f'Missing required fields: {missing}')
    nd, nt = _dimension(fold['nD'], 'nD'), _dimension(fold['nT'], 'nT')
    mask = np.asarray(fold['train_pos_mask'])
    if mask.shape != (nd, nt) or not np.isin(mask, [0, 1]).all():
        raise ValueError('train_pos_mask must be a binary drug-by-target matrix')
    seen = set()
    arrays = {}
    for name in ('train_pos', 'train_neg', 'val_pos', 'val_neg', 'test_pos', 'test_neg'):
        if name not in fold:
            continue
        pairs = _pairs(fold[name], name, nd, nt)
        identities = set(map(tuple, pairs.tolist()))
        if identities & seen:
            raise AssertionError(f'{name} overlaps another split or label class')
        seen.update(identities)
        arrays[name] = pairs
    expected = np.zeros((nd, nt), dtype=bool)
    expected[tuple(arrays['train_pos'].T)] = True
    if not np.array_equal(mask.astype(bool), expected):
        raise AssertionError('train_pos_mask must contain exactly training positives')
    return True


def load_fold(root, dataset, k):
    """Load numeric arrays without pickle and close the archive before returning."""
    path = os.path.join(root, dataset, f'fold_{k}.npz')
    with np.load(path, allow_pickle=False) as source:
        fold = {key: source[key] for key in source.files}
    check_no_leakage(fold)
    return fold
