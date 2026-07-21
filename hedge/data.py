# -*- coding: utf-8 -*-
"""Data interface.

This release deliberately ships only the *interface*, not the dataset
construction pipeline. Any heterogeneous DTI benchmark can be consumed by
writing one ``fold_{k}.npz`` per cross-validation fold with the fields below.

Required fields
---------------
nD              int                   number of drugs
nT              int                   number of targets
train_pos_mask  (nD, nT) {0,1}        TRAIN-SPLIT POSITIVES ONLY.
                                      Held-out edges must never appear here --
                                      this array is what the sampler walks over,
                                      so any test edge left in it is leakage.
train_pos       (P, 2) int            (drug, target) training positives
test_pos        (Q, 2) int            (drug, target) held-out positives
test_neg        (R, 2) int            (drug, target) sampled negatives

Evaluation protocol
-------------------
Scores are written as a dense (nD, nT) matrix per fold and ranked over the FULL
target space, excluding training positives. Precision@K computed over a sampled
candidate shortlist is not comparable to Precision@K over the full space, and we
do not report the former.
"""
import os
import numpy as np

REQUIRED = ("nD", "nT", "train_pos_mask", "train_pos", "test_pos", "test_neg")


def load_fold(root, dataset, k):
    """Load fold k of a dataset directory; validates the interface contract."""
    path = os.path.join(root, dataset, f"fold_{k}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    fold = np.load(path, allow_pickle=True)
    missing = [f for f in REQUIRED if f not in fold]
    if missing:
        raise KeyError(f"{path} is missing required field(s): {missing}")
    return fold


def check_no_leakage(fold):
    """Assert that no held-out positive is present in the sampling graph."""
    mask = fold["train_pos_mask"]
    tp = fold["test_pos"]
    hit = int(sum(1 for d, t in tp if mask[int(d), int(t)] != 0))
    if hit:
        raise AssertionError(
            f"{hit} held-out positives found in train_pos_mask -- the sampling "
            f"graph is leaking test edges.")
    return True
