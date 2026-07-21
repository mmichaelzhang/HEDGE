# -*- coding: utf-8 -*-
"""Dual-view training objective  (paper Sec. 4.2.2).

Two complementary views of every sample:
  * a CONTRASTIVE view over head--tail pairs   -> L_con
  * an OBSERVATIONAL view over connections     -> L_obs
plus a per-drug LISTWISE ranking term that targets Precision@K directly, which
is the quantity that governs wet-lab prioritisation.

    L = L_bce + lam_list * L_list + lam_con * L_con + lam_obs * L_obs

All four terms are kept on a comparable scale. In particular the observational
view is computed on L2-normalised representations with a temperature, exactly
like the contrastive view: on raw d-dimensional representations the dot product
is O(sqrt(d)) (~11 at d=128, excursions past 30), which makes L_obs roughly an
order of magnitude larger than the other three and lets it dominate the
objective regardless of its weight.
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
