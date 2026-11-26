"""Regression tests for the legacy public interface, with synthetic inputs only."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from hedge.data import check_no_leakage, load_fold
from hedge.sampling import build_typed_graph, sample_walks


def fixture():
    return dict(nD=np.array(2), nT=np.array(3),
                train_pos_mask=np.array([[1, 0, 0], [0, 1, 0]]),
                train_pos=np.array([[0, 0], [1, 1]]),
                val_pos=np.array([[0, 1]]), test_pos=np.array([[1, 2]]),
                test_neg=np.array([[0, 2]]))


class Contracts(unittest.TestCase):
    def test_tiny_graph_default_topk_and_no_same_type_edges_at_zero(self):
        mask = np.ones((2, 3), dtype=int)
        off, nbr, typ, n = build_typed_graph(mask)
        self.assertEqual(n, 5)
        self.assertTrue((nbr[:off[-1]] < n).all())
        for a in range(n):
            self.assertNotIn(a, nbr[off[a]:off[a+1]])
        off, _, typ, _ = build_typed_graph(mask, topk=0)
        np.testing.assert_array_equal(typ[:off[-1]], 0)
        with self.assertRaises(ValueError):
            build_typed_graph(mask, topk=-1)

    def test_dead_end_walk(self):
        off, nbr, typ, _ = build_typed_graph(np.zeros((1, 1), dtype=int))
        nodes, relations = sample_walks(np.array([0]), off, nbr, typ, 2, 3, np.random.default_rng(0))
        np.testing.assert_array_equal(nodes, np.zeros((2, 3), dtype=int))
        self.assertEqual(relations.shape, (2, 2))

    def test_exact_graph_mask_and_validation_leak_rejected(self):
        value = fixture()
        self.assertTrue(check_no_leakage(value))
        value['train_pos_mask'][0, 1] = 1
        with self.assertRaises(AssertionError):
            check_no_leakage(value)
        value = fixture()
        value['test_pos'] = value['train_pos'][:1]
        with self.assertRaises(AssertionError):
            check_no_leakage(value)

    def test_noninteger_and_out_of_range_ids_rejected(self):
        for pairs in [np.array([[0., 1.5]]), np.array([[2, 0]])]:
            value = fixture(); value['test_pos'] = pairs
            with self.assertRaises(ValueError):
                check_no_leakage(value)

    def test_npz_loader_closes_file_and_rejects_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'toy'; dest.mkdir()
            p = dest/'fold_0.npz'
            np.savez(p, **fixture())
            self.assertIsInstance(load_fold(tmp, 'toy', 0), dict)
            np.savez(p, **fixture(), unsafe=np.array([{}], dtype=object))
            with self.assertRaises(ValueError):
                load_fold(tmp, 'toy', 0)


if __name__ == '__main__':
    unittest.main()
