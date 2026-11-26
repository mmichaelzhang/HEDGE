# -*- coding: utf-8 -*-
"""Legacy demonstration losses; numerical behavior is retained.

These include hard-negative-weighted BCE, a listwise surrogate, contrastive
loss, and random-negative observation loss. They are not the later audited
attention-weighted triple observation objective in reference/paper_core.py.
"""
import torch
import torch.nn.functional as F


def bce_with_hard_negatives(logit, y, hardneg=True):
    """Binary term; confidently-wrong negatives are up-weighted."""
    w = torch.ones_like(y)
    if hardneg:
        with torch.no_grad():
            hard = (torch.sigmoid(logit) > 0.5) & (y == 0)
        w = w + hard.float()
    return (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean()


def listwise_ranking(logit, y, drow, n_heads):
    """Per-drug listwise softmax over that drug's candidate tails (Precision@K)."""
    loss = torch.zeros((), device=logit.device)
    for r in range(n_heads):
        sel = (drow == r)
        if sel.sum() < 2 or y[sel].sum() == 0:
            continue
        lp = F.log_softmax(logit[sel], 0)
        loss = loss - (lp * (y[sel] / y[sel].sum())).sum()
    return loss / max(n_heads, 1)


def contrastive_infonce(hd, ht, y, drow, n_heads, tau=0.2):
    """Supervised InfoNCE: head anchor, interacting tails positive (Eq. 5)."""
    sim = (F.normalize(hd, dim=-1) * F.normalize(ht, dim=-1)).sum(-1) / tau
    loss = torch.zeros((), device=sim.device)
    for r in range(n_heads):
        sel = (drow == r)
        if sel.sum() < 2 or y[sel].sum() == 0:
            continue
        loss = loss - (F.log_softmax(sim[sel], 0) * (y[sel] / y[sel].sum())).sum()
    return loss / max(n_heads, 1)


def observation(H, a_idx, b_idx, neg_idx, tau=0.5, normalize=True):
    """Connection-attentive observation likelihood (Eq. 7).

    Nodes joined by a sampled connection should agree; a random node should not.
    Scores are L2-normalised and temperature-scaled to keep this term on the same
    scale as the contrastive view (see module docstring).
    """
    if normalize:
        Hn = F.normalize(H, dim=-1)
        pos = (Hn[a_idx] * Hn[b_idx]).sum(-1) / tau
        neg = (Hn[a_idx] * Hn[neg_idx]).sum(-1) / tau
    else:  # legacy unnormalised behaviour, retained for ablation
        pos = (H[a_idx] * H[b_idx]).sum(-1)
        neg = (H[a_idx] * H[neg_idx]).sum(-1)
    return -F.logsigmoid(pos).mean() - F.logsigmoid(-neg).mean()
