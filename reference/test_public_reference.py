"""Additional contracts around the portable, source-bound entry points."""
import copy
import contextlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from paper_core import collate
from main_components import MainCore, BlockPairSchedule, query_block_labels, objective
from support_query_contract import RealViewIndex, query_rng, filter_dti_csr, pair_mask
from support_query_runtime import PackedContext, plan_view_ids, streaming_score
from stable_ensemble_score import score
from synthetic_example import fixture, run
from verify_sources import verify


class PublicReferenceContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_frozen_source_bytes(self):
        self.assertEqual(verify()['status'], 'PASS')

    def test_constant_tuple_same_initial_state_and_zero_data_gradient(self):
        samples, vocab, _, _, _ = fixture()
        torch.manual_seed(4)
        a = MainCore(6, len(vocab), 2, d=8, nheads=2, nlayers=1, dropout=0)
        torch.manual_seed(4)
        b = MainCore(6, len(vocab), 2, d=8, nheads=2, nlayers=1, dropout=0, constant_tuple=True)
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
        batch = collate(samples, vocab)
        before = b.encode(batch).nodes
        with torch.no_grad():
            b.connection_embedding.weight.add_(100)
        after = b.encode(batch).nodes
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        after.square().sum().backward()
        self.assertEqual(float(b.connection_embedding.weight.grad.abs().sum()), 0)

    def test_objective_uses_current_explicit_labels_and_all_query_branches(self):
        samples, vocab, block, pairs, labels = fixture()
        model = MainCore(6, len(vocab), 2, d=8, nheads=2, nlayers=1, dropout=0)
        index = RealViewIndex([s.nodes for s in samples], 2, 4)
        plans = [index.plan(int(h), int(t)+2, query_rng(0, 0, i, int(h), int(t)+2))
                 for i, (h, t) in enumerate(pairs)]
        self.assertEqual({p.branch for p in plans}, {'cooccurrence','global_context','input_target'})
        ids = plan_view_ids(plans)
        encoded = model.encode(collate([samples[int(i)] for i in ids], vocab))
        loss, parts, stats = objective(model, encoded, plans, ids, labels,
            query_block_labels(block, 2, 4), 2, 1., 'hidden_full')
        torch.testing.assert_close(loss, parts[0]+.5*parts[1]+.3*parts[2])
        self.assertEqual(stats['current_block_negative_tail_groups'], 0)
        # Unlabelled target 3 is context, not an invented negative example.
        loss.backward()
        self.assertTrue(torch.isfinite(model.predictor[0].weight.grad).all())

    def test_streaming_matches_finite_bank_score_and_fallback_counts(self):
        samples, vocab, _, _, _ = fixture()
        batch = collate(samples, vocab)
        model = MainCore(6, len(vocab), 2, d=8, nheads=2, nlayers=1, dropout=0).eval()
        expected, counts, _ = score(model, batch, 2, 4, bs=2)
        with contextlib.ExitStack() as stack:
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            arrays = {'nodes': batch.nodes.numpy().astype(np.int16),
                      'connections': batch.connections.numpy().astype(np.uint16),
                      'distances': batch.distances.numpy().astype(np.uint8),
                      'lengths': batch.valid.sum(1).numpy().astype(np.uint8)}
            for name, arr in arrays.items():
                np.save(tmp/(name+'.npy'), arr)
            context = PackedContext(tmp, 2, 4, 2)
            stack.callback(context.close)
            actual, n, _ = streaming_score(model, context, bs=2)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(n, counts)
            self.assertEqual(int(n[0, 2]), 0)  # target 4: global-context fallback
            self.assertEqual(int(n[0, 3]), 0)  # target 5: input-target fallback

    def test_five_block_schedule_and_bidirectional_hiding(self):
        schedule = BlockPairSchedule([3]*5, fold=2, batch_size=2)
        seen = [[] for _ in range(5)]
        for update in range(10):
            block, indices = schedule.next()
            self.assertEqual(block, update % 5)
            seen[block].extend(indices.tolist())
        for values in seen:
            self.assertEqual(sorted(values), [0, 1, 2])
        # Drug0<->target2 and drug1<->target3, plus an auxiliary edge.
        off = np.array([0, 2, 3, 4, 5, 5])
        nb = np.array([2, 4, 3, 0, 1]); rel = np.array([0, 3, 0, 0, 0])
        support = pair_mask(np.array([[1, 1]]), 2, 2)
        a, b, c = filter_dti_csr(off, nb, rel, support)
        edges = list(zip(np.repeat(np.arange(5), np.diff(a)), b, c))
        self.assertEqual(edges, [(0, 4, 3), (1, 3, 0), (3, 1, 0)])

    def test_synthetic_example(self):
        result = run()
        self.assertTrue(result['gradients_finite'])
        self.assertEqual(result['score_shape'], [2, 4])


if __name__ == '__main__':
    unittest.main()
