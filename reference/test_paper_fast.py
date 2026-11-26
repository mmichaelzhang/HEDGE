"""Value and gradient equivalence against the explicit paper_core definitions."""
import unittest
import torch
import paper_core as core
import paper_fast as fast


class FastEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(117)
        self.samples = [core.Sample((0, 2, 2, 3), ((0,), (1,), (2,)), (0, 1, 1, 2)),
                        core.Sample((1, 3, 4), ((0,), (1,)), (0, 1, 2)),
                        core.Sample((0, 2), ((0,),), (0, 1)),
                        core.Sample((1,), (), (0,))]
        self.vocab = core.TupleVocabulary.from_samples(self.samples)
        self.batch = core.collate(self.samples, self.vocab)
        self.positive = {0: {2}, 1: {4}}
        self.model = core.PaperCore(7, len(self.vocab), 5, d=8, nheads=2, dropout=0).eval()

    def test_losses_and_gradients_with_duplicate_targets_and_cold_filter(self):
        for normalize in [False, True]:
            for allowed in [None, {2, 3}]:
                a = torch.randn(4, 4, 8, requires_grad=True)
                b = a.detach().clone().requires_grad_(True)
                ea = core.Encoded(self.batch, a, torch.randn(4, 3, 8))
                eb = core.Encoded(self.batch, b, ea.connections)
                index = fast.TailIndex.from_batch(self.batch, self.positive, allowed)
                la = core.contrastive_loss(ea, self.positive, normalize=normalize, allowed_targets=allowed)
                lb = fast.contrastive_loss(eb, index, normalize=normalize)
                self.assertTrue(torch.allclose(la, lb, atol=1e-6), (la, lb))
                la.backward()
                lb.backward()
                self.assertTrue(torch.allclose(a.grad, b.grad, atol=2e-6))
        ea = self.model.encode(self.batch)
        index = fast.TailIndex.from_batch(self.batch, self.positive)
        a = core.decoder_bce(self.model, ea, self.positive, pos_weight=3)
        b = fast.decoder_bce(self.model, ea, index, pos_weight=3)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_streaming_all_scores_counts_and_fallback_exact(self):
        with torch.no_grad():
            e1 = self.model.encode(core.collate(self.samples[:2], self.vocab))
            e2 = self.model.encode(core.collate(self.samples[2:], self.vocab))
            fallback = {5: torch.randn(8)}
            reference = core.ensemble_probabilities(self.model, [e1, e2], [0, 1], [2, 3, 4, 5],
                fallback="global_target_mean", fallback_target_states=fallback)
            accumulator = fast.StreamingEnsemble(self.model, [0, 1], [2, 3, 4, 5])
            accumulator.add(e1)
            accumulator.add(e2)
            result = accumulator.finalize(fallback="global_target_mean",
                                         fallback_target_states=fallback, head_chunk=1, target_chunk=2)
            self.assertTrue(torch.allclose(reference[0], result[0], atol=1e-6))
            self.assertTrue(torch.equal(reference[1], result[1]))
            self.assertTrue(torch.equal(reference[2], result[2]))
            self.assertEqual(int(result[2][0, 2]), 1)  # target4 seen in other head's view
            self.assertEqual(int(result[2][0, 3]), 2)  # target5 absent from all views
            with self.assertRaises(ValueError):
                accumulator.finalize()

    def test_empty_candidate_losses_are_finite_zero(self):
        encoded = self.model.encode(self.batch)
        index = fast.TailIndex.from_batch(self.batch, self.positive, allowed_targets=set())
        self.assertEqual(float(fast.contrastive_loss(encoded, index).detach()), 0)
        self.assertEqual(float(fast.decoder_bce(self.model, encoded, index).detach()), 0)


if __name__ == "__main__":
    unittest.main()
