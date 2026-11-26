"""CPU-only example of the source-bound core; no benchmark data or metrics.

The hand-written views exercise co-occurrence, global-context and input-target
branches. They are illustrative, not a replacement for audited bank preparation.
"""
import json
import numpy as np
import torch
from paper_core import Sample, TupleVocabulary, collate
from main_components import MainCore, objective, query_block_labels
from support_query_contract import RealViewIndex, query_rng
from support_query_runtime import plan_view_ids
from stable_ensemble_score import score


def fixture():
    samples = [Sample((0, 2, 2, 3), ((0,), (2,), (1, 3)), (0, 1, 1, 2)),
               Sample((0, 2), ((0,),), (0, 1)),
               Sample((1, 4), ((0,),), (0, 1)),
               Sample((1,), (), (0,))]
    vocabulary = TupleVocabulary.from_samples(samples)
    block = {'positive': np.array([[0, 0], [1, 2]], dtype=np.int64),
             'negative': np.array([[0, 2], [1, 3]], dtype=np.int64)}
    pairs = np.concatenate((block['positive'], block['negative']))
    labels = np.array([1, 1, 0, 0], dtype=np.float32)
    return samples, vocabulary, block, pairs, labels


def run():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    samples, vocabulary, block, pairs, labels = fixture()
    model = MainCore(6, len(vocabulary), 2, d=8, nheads=2, nlayers=1, dropout=0)
    index = RealViewIndex([s.nodes for s in samples], 2, 4)
    plans = [index.plan(int(h), int(t)+2, query_rng(0, 0, i, int(h), int(t)+2))
             for i, (h, t) in enumerate(pairs)]
    ids = plan_view_ids(plans)
    encoded = model.encode(collate([samples[int(i)] for i in ids], vocabulary))
    loss, parts, exposure = objective(model, encoded, plans, ids, labels,
        query_block_labels(block, 2, 4), 2, 1., 'hidden_full')
    loss.backward()
    gradients_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                           for p in model.parameters())
    # One synthetic optimizer step; no experiment checkpoint selection occurs.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    matrix, counts, timing = score(model, collate(samples, vocabulary), 2, 4, bs=2)
    assert np.isfinite(matrix).all() and gradients_finite
    return {'scope': 'synthetic CPU example; no benchmark metrics',
            'optimizer_steps': 1, 'gradients_finite': gradients_finite,
            'score_shape': list(matrix.shape), 'score_type': 'log_odds_of_mean_view_probability',
            'branches': sorted({p.branch for p in plans}),
            'view_counts': counts.tolist(), 'auxiliary_exposure': exposure}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
