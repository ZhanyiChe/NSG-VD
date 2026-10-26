import unittest

import torch

from utils.adaptive_evaluation import (
    AdaptiveReferencePool,
    select_and_score_adaptive_batch,
)
from utils.adaptive_mmd import (
    precompute_candidate_kernel_matrix,
    slow_adaptive_mmd_scores,
)


class AdaptiveEvaluationTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.candidate_deep = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        self.candidate_raw = torch.randn(5, 2, 2, generator=generator, dtype=torch.float64)
        self.query_deep = torch.randn(1, 3, generator=generator, dtype=torch.float64)
        self.query_raw = torch.randn(1, 2, 2, generator=generator, dtype=torch.float64)
        self.sigma = 2.0
        self.sigma0 = 0.5
        self.epsilon = 0.1
        kernel = precompute_candidate_kernel_matrix(
            self.candidate_deep,
            self.candidate_raw,
            self.sigma,
            self.sigma0,
            self.epsilon,
            block_size=2,
        )
        clip_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        self.pool = AdaptiveReferencePool(
            video_ids=("A", "B", "C", "D", "E"),
            raw_features=self.candidate_raw,
            deep_features=self.candidate_deep,
            kernel_matrix=kernel,
            clip_embeddings=clip_embeddings,
        )

    def test_clip_selection_flows_into_exact_mmd_score(self):
        result = select_and_score_adaptive_batch(
            self.pool,
            query_ids=["query"],
            query_raw_features=self.query_raw,
            query_deep_features=self.query_deep,
            mode="clip_topn",
            top_n=2,
            random_seed=42,
            sigma=self.sigma,
            sigma0=self.sigma0,
            epsilon=self.epsilon,
            is_smooth=True,
            query_clip_embeddings=torch.tensor([[1.0, 0.0]]),
        )

        self.assertEqual(
            [selection.reference_id for selection in result.selections[0]],
            ["A", "B"],
        )
        expected = slow_adaptive_mmd_scores(
            self.candidate_deep,
            self.candidate_raw,
            self.query_deep,
            self.query_raw,
            torch.tensor([[0, 1]]),
            self.sigma,
            self.sigma0,
            self.epsilon,
        )
        torch.testing.assert_close(result.scores, expected, rtol=1e-12, atol=1e-12)

    def test_random_selection_is_independent_of_batch_repetition(self):
        arguments = dict(
            reference_pool=self.pool,
            query_ids=["query"],
            query_raw_features=self.query_raw,
            query_deep_features=self.query_deep,
            mode="random_topn",
            top_n=3,
            random_seed=9,
            sigma=self.sigma,
            sigma0=self.sigma0,
            epsilon=self.epsilon,
            is_smooth=True,
        )

        first = select_and_score_adaptive_batch(**arguments)
        second = select_and_score_adaptive_batch(**arguments)

        self.assertEqual(first.selections, second.selections)
        torch.testing.assert_close(first.scores, second.scores)

    def test_clip_mode_requires_both_candidate_and_query_embeddings(self):
        pool_without_clip = AdaptiveReferencePool(
            video_ids=self.pool.video_ids,
            raw_features=self.pool.raw_features,
            deep_features=self.pool.deep_features,
            kernel_matrix=self.pool.kernel_matrix,
        )
        with self.assertRaisesRegex(ValueError, "candidate CLIP embeddings"):
            select_and_score_adaptive_batch(
                pool_without_clip,
                ["query"],
                self.query_raw,
                self.query_deep,
                "clip_topn",
                2,
                0,
                self.sigma,
                self.sigma0,
                self.epsilon,
                True,
                query_clip_embeddings=torch.tensor([[1.0, 0.0]]),
            )


if __name__ == "__main__":
    unittest.main()
