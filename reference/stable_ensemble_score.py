"""Numerically stable ranking score for mean-probability view ensembling.

score = log(sum_view sigmoid(logit)) - log(sum_view sigmoid(-logit)).
The view count cancels: this is logit(mean_view sigmoid(logit)), strictly
monotonic in the original ensemble probability. Accumulating both sides in log
space preserves ordering near probabilities zero AND one without clipping.
"""
import numpy as np
import torch
from torch.nn import functional as F
from paper_core import Batch
from paper_fast import TailIndex

SCORE_TYPE = 'log_odds_of_mean_view_probability'


def _group_logsum(values, inverse, groups):
    maximum = values.new_full((groups,), -torch.inf)
    maximum.scatter_reduce_(0, inverse, values, reduce='amax', include_self=True)
    total = values.new_zeros(groups).index_add_(0, inverse, torch.exp(values - maximum[inverse]))
    return maximum + torch.log(total)


@torch.no_grad()
def score(model, batch, nd, nt, bs=512, head_chunk=16, target_chunk=512):
    """Return finite ranking log-odds, view counts and coverage diagnostics.

    Graph encoding, within-view repeated-target averaging, equal view weights,
    mean-context fallback, and globally absent target inputs match the original
    runner. Only numerical accumulation/output score representation differs.
    Do not label the returned real-valued array a calibrated probability.
    """
    model.eval()
    device = batch.nodes.device
    hs = torch.zeros(nd, model.d, device=device)
    hc = torch.zeros(nd, device=device)
    ts = torch.zeros(nt, model.d, device=device)
    tc = torch.zeros(nt, device=device)
    log_p = torch.full((nd * nt,), -torch.inf, dtype=torch.float64, device=device)
    log_q = torch.full_like(log_p, -torch.inf)
    counts = torch.zeros(nd * nt, dtype=torch.long, device=device)
    for start in range(0, len(batch.nodes), bs):
        b = Batch(*(getattr(batch, key)[start:start + bs]
                    for key in ['nodes', 'connections', 'distances', 'valid']))
        encoded = model.encode(b)
        heads = b.nodes[:, 0]
        hs.index_add_(0, heads, encoded.nodes[:, 0])
        hc.index_add_(0, heads, torch.ones(len(heads), device=device))
        index = TailIndex.from_batch(b)
        if not len(index.row):
            continue
        means = index.mean_states(encoded)
        targets = index.target - nd
        if (targets < 0).any() or (targets >= nt).any():
            raise ValueError('Expected retained tails to be targets in the supplied universe')
        logits = model.pair_logit(encoded.nodes[index.row, 0], means).double()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite pair logits; cannot repair them by score clipping')
        pair = heads[index.row] * nt + targets
        unique, inverse = torch.unique(pair, sorted=True, return_inverse=True)
        chunk_p = _group_logsum(F.logsigmoid(logits), inverse, len(unique))
        chunk_q = _group_logsum(F.logsigmoid(-logits), inverse, len(unique))
        log_p[unique] = torch.logaddexp(log_p[unique], chunk_p)
        log_q[unique] = torch.logaddexp(log_q[unique], chunk_q)
        counts.index_add_(0, pair, torch.ones_like(pair))
        ts.index_add_(0, targets, means)
        tc.index_add_(0, targets, torch.ones(len(targets), device=device))
    if (hc == 0).any():
        raise ValueError('Every evaluated drug requires at least one encoded view')
    hm = hs / hc[:, None]
    tm = ts / tc.clamp_min(1)[:, None]
    missing = tc == 0
    if missing.any():
        tm[missing] = model.node_input(torch.arange(nd, nd + nt, device=device)[missing])
    counts = counts.view(nd, nt)
    log_p, log_q = log_p.view(nd, nt), log_q.view(nd, nt)
    result = torch.empty(nd, nt, dtype=torch.float64, device=device)
    for i in range(0, nd, head_chunk):
        for j in range(0, nt, target_chunk):
            fallback_logit = model.pair_logit(hm[i:i + head_chunk, None],
                                              tm[None, j:j + target_chunk]).double()
            c = counts[i:i + head_chunk, j:j + target_chunk]
            observed_log_odds = log_p[i:i + head_chunk, j:j + target_chunk] - log_q[i:i + head_chunk, j:j + target_chunk]
            # For a one-score fallback, logit(sigmoid(raw_logit)) is raw_logit.
            result[i:i + head_chunk, j:j + target_chunk] = torch.where(c > 0, observed_log_odds, fallback_logit)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite ensemble score; inspect model states/logits')
    coverage = {'global_absent_targets': int(missing.sum()),
                'cooccurrence_pair_fraction': float((counts > 0).float().mean()),
                'input_only_fallback_pair_fraction': float(((counts == 0) & missing[None]).float().mean())}
    return result.cpu().numpy(), counts.cpu().numpy(), coverage
