import unittest

import numpy as np
import torch

from test_and_evaluate import evaluate, evaluate_retrieval


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.query = torch.nn.functional.normalize(torch.randn(7, 8), dim=1)
        self.gallery = torch.nn.functional.normalize(torch.randn(11, 8), dim=1)
        self.query_labels = np.array([0, 1, 2, 0, 1, 2, 3])
        self.gallery_labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2])

    def test_batched_matches_single_query_for_all_distances(self):
        for distance in ("Cos", "Eu", "Man"):
            expected = [
                evaluate(self.query[i], label, self.gallery, self.gallery_labels, distance)
                for i, label in enumerate(self.query_labels)
            ]
            cmc, ap, valid = evaluate_retrieval(
                self.query,
                self.query_labels,
                self.gallery,
                self.gallery_labels,
                dist=distance,
                chunk_size=3,
                device="cpu",
            )
            self.assertEqual(valid, len(self.query_labels))
            np.testing.assert_allclose(cmc, np.mean([item[1].numpy() for item in expected], axis=0))
            self.assertAlmostEqual(ap, np.mean([item[0] for item in expected]))

    def test_invalid_distance_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval(
                self.query,
                self.query_labels,
                self.gallery,
                self.gallery_labels,
                dist="invalid",
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
