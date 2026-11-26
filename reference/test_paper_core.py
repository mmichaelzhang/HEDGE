"""Scientific-invariant checks; run: python -m unittest test_paper_core -v."""
import unittest
import numpy as np
import torch
from torch import nn
from paper_core import (TrainGraph, Sample, TupleVocabulary, collate, Encoded,
                        PaperCore, contrastive_loss, decoder_bce,
                        ensemble_probabilities, assert_train_only_dti)


def graph(edges, n=7, nd=2, nt=3):
    adj = [[] for _ in range(n)]
    for u, v, relation in edges:
        adj[u].append((v, relation))
    offsets, neighbors, relations = [0], [], []
    for row in adj:
        for v, rel in row:
            neighbors.append(v)
            relations.append(rel)
        offsets.append(len(neighbors))
    return TrainGraph(offsets, neighbors, relations, nd, nt)


class InvariantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_single_head_auxiliary_drug_order_and_exact_distance(self):
        # Deterministic walk: head0 -> drug1 -> aux5 -> target2 -> aux6 -> target3.
        g = graph([(0, 1, 1), (1, 5, 3), (5, 2, 4), (2, 6, 4), (6, 3, 4)])
        s = g.sample([0], 1, 3, 2, np.random.default_rng(0))[0]
        self.assertEqual(s.nodes, (0, 2, 3))
        self.assertEqual(s.connections, ((1, 3, 4), (4, 4)))
        self.assertEqual(s.distances, (0, 3, 5))
        # Exceeding gap budget stops, and does not leave drug/aux tokens behind.
        stopped = g.sample([0], 1, 3, 1, np.random.default_rng(0))[0]
        self.assertEqual(stopped.nodes, (0,))
        self.assertEqual(stopped.connections, ())

    def test_position_is_graph_distance_not_walk_index(self):
        g = graph([(0, 2, 0), (0, 5, 3), (5, 6, 1), (6, 2, 4)])
        self.assertEqual(g.shortest_paths(0)[2], 1)
        # Target2 can appear after a long connection or direct connection;
        # its positional encoding is one in either case.
        samples = g.sample([0], 40, 2, 2, np.random.default_rng(7))
        self.assertEqual({s.connections[0] for s in samples}, {(0,), (3, 1, 4)})
        self.assertTrue(all(s.distances == (0, 1) for s in samples))

    def test_tuple_order_and_multiplicity_never_alias(self):
        vocab = TupleVocabulary([(1, 2), (2, 1), (1,), (1, 1)])
        self.assertEqual(len({vocab.ids[x] for x in [(1, 2), (2, 1), (1,), (1, 1)]}), 4)
        with self.assertRaises(KeyError):
            collate([Sample((0, 2), ((9,),), (0, 1))], vocab)

    def test_normalized_observation_bounded_and_padding_safe(self):
        samples = [Sample((0, 2, 3), ((0,), (2,)), (0, 1, 2)), Sample((1,), (), (0,))]
        batch = collate(samples, TupleVocabulary.from_samples(samples))
        model = PaperCore(7, 3, 5, d=4, nheads=1, dropout=0)
        h = torch.tensor([[[1., 0, 0, 0], [1., 1, 0, 0], [0., 1, 0, 0]],
                          [[1., 0, 0, 0], [0., 0, 0, 0], [0., 0, 0, 0]]])
        e = Encoded(batch, h, torch.zeros(2, 2, 4))
        # Uniform attention isolates the agreement scale property.
        with torch.no_grad():
            model.attention_vector.zero_()
        normalized, alpha = model.observation_loss(e, normalize=True)
        scaled = Encoded(batch, 20 * h, e.connections)
        again, _ = model.observation_loss(scaled, normalize=True)
        raw, _ = model.observation_loss(e, normalize=False)
        raw_scaled, _ = model.observation_loss(scaled, normalize=False)
        self.assertTrue(torch.allclose(normalized, again))
        self.assertTrue(torch.allclose(raw_scaled, raw * 400))
        self.assertTrue(torch.allclose(alpha.sum(1), torch.tensor([1., 0.])))
        self.assertTrue(torch.isfinite(normalized))
        self.assertLessEqual(abs(float(normalized.detach())), 1)

    def test_pad_does_not_change_valid_encoder_states(self):
        short = Sample((0, 2), ((0,),), (0, 1))
        long = Sample((1, 3, 4), ((0,), (2,)), (0, 1, 2))
        vocab = TupleVocabulary.from_samples([short, long])
        model = PaperCore(7, len(vocab), 5, d=8, nheads=2, dropout=0).eval()
        a = model.encode(collate([short], vocab)).nodes[0]
        b = model.encode(collate([short, long], vocab)).nodes[0, :2]
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_ensemble_equal_weight_per_view_and_probability_mean(self):
        samples = [Sample((0, 2, 2), ((0,), (2,)), (0, 1, 1)),
                   Sample((0, 2), ((0,),), (0, 1))]
        batch = collate(samples, TupleVocabulary.from_samples(samples))
        # First view repeats target but should count once: logits 0 and 4.
        encoded = Encoded(batch, torch.tensor([[[1.], [0.], [0.]],
                                               [[1.], [4.], [0.]]]), torch.zeros(2, 2, 1))
        class Scorer:
            def pair_logit(self, head, target):
                return (head * target).sum(-1)
        result, counts, kinds = ensemble_probabilities(Scorer(), [encoded], [0], [2])
        expected = (torch.sigmoid(torch.tensor(0.)) + torch.sigmoid(torch.tensor(4.))) / 2
        self.assertAlmostEqual(float(result), float(expected), places=6)
        self.assertEqual(int(counts), 2)
        self.assertEqual(int(kinds), 0)
        self.assertFalse(torch.allclose(result, torch.sigmoid(torch.tensor(2.))))
        with self.assertRaises(ValueError):
            ensemble_probabilities(Scorer(), [encoded], [0], [3])
        result, counts, kinds = ensemble_probabilities(
            Scorer(), [encoded], [0], [3], fallback="global_target_mean",
            fallback_target_states={3: torch.tensor([2.])})
        self.assertEqual(int(kinds), 2)
        self.assertEqual(int(counts), 0)
        self.assertAlmostEqual(float(result), float(torch.sigmoid(torch.tensor(2.))), places=6)

    def test_decoder_needs_explicit_supervision(self):
        samples = [Sample((0, 2, 3), ((0,), (2,)), (0, 1, 2))]
        vocab = TupleVocabulary.from_samples(samples)
        model = PaperCore(7, len(vocab), 5, d=8, nheads=2, dropout=0)
        encoded = model.encode(collate(samples, vocab))
        loss = contrastive_loss(encoded, {0: {2}}) + model.observation_loss(encoded)[0]
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.predictor.parameters()))
        model.zero_grad(set_to_none=True)
        encoded = model.encode(collate(samples, vocab))
        decoder_bce(model, encoded, {0: {2}}).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.predictor.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.predictor.parameters()), 0)

    def test_excluded_cold_target_does_not_enter_contrastive_denominator(self):
        samples = [Sample((0, 2, 3), ((0,), (2,)), (0, 1, 2))]
        batch = collate(samples, TupleVocabulary.from_samples(samples))
        # An excluded target with a huge score would otherwise alter the loss.
        encoded = Encoded(batch, torch.tensor([[[1.], [0.], [100.]]]), torch.zeros(1, 2, 1))
        excluded = contrastive_loss(encoded, {0: {2}}, allowed_targets={2})
        included = contrastive_loss(encoded, {0: {2}})
        self.assertEqual(float(excluded), 0.)
        self.assertGreater(float(included), 100.)

    def test_graph_leakage_guard_checks_reverse_edges(self):
        clean = graph([(0, 2, 0), (2, 0, 0), (0, 5, 3)])
        assert_train_only_dti(clean, [(0, 0)], [(1, 1)])
        leaked = graph([(0, 2, 0), (2, 0, 0), (3, 1, 0)])
        with self.assertRaises(AssertionError):
            assert_train_only_dti(leaked, [(0, 0)], [(1, 1)])


if __name__ == "__main__":
    unittest.main()
