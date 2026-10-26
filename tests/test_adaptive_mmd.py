import unittest

import torch

from utils.adaptive_mmd import (
    fast_adaptive_mmd_scores,
    precompute_candidate_kernel_matrix,
    slow_adaptive_mmd_scores,
)
from utils.mmd_utils import MMD_batch2


class AdaptiveMMDTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.candidate_deep = torch.randn(7, 4, generator=generator, dtype=torch.float64)
        self.candidate_raw = torch.randn(7, 2, 3, generator=generator, dtype=torch.float64)
        self.query_deep = torch.randn(3, 4, generator=generator, dtype=torch.float64)
        self.query_raw = torch.randn(3, 2, 3, generator=generator, dtype=torch.float64)
        self.indices = torch.tensor(
            [[0, 2, 4, 6], [1, 2, 3, 5], [0, 3, 5, 6]], dtype=torch.long
        )
        self.sigma = torch.tensor(2.3, dtype=torch.float64)
        self.sigma0 = torch.tensor(0.7, dtype=torch.float64)
        self.epsilon = torch.tensor(0.12, dtype=torch.float64)

    def _compare_fast_and_slow(self, is_smooth):
        candidate_kernel = precompute_candidate_kernel_matrix(
            self.candidate_deep,
            self.candidate_raw,
            self.sigma,
            self.sigma0,
            self.epsilon,
            is_smooth=is_smooth,
            block_size=3,
        )
        slow = slow_adaptive_mmd_scores(
            self.candidate_deep,
            self.candidate_raw,
            self.query_deep,
            self.query_raw,
            self.indices,
            self.sigma,
            self.sigma0,
            self.epsilon,
            is_smooth=is_smooth,
        )
        fast = fast_adaptive_mmd_scores(
            self.candidate_deep,
            self.candidate_raw,
            self.query_deep,
            self.query_raw,
            self.indices,
            candidate_kernel,
            self.sigma,
            self.sigma0,
            self.epsilon,
            is_smooth=is_smooth,
        )
        torch.testing.assert_close(fast, slow, rtol=1e-12, atol=1e-12)

    def test_fast_matches_original_mmd_batch2_for_smooth_kernel(self):
        self._compare_fast_and_slow(is_smooth=True)

    def test_fast_matches_original_mmd_batch2_for_deep_only_kernel(self):
        self._compare_fast_and_slow(is_smooth=False)

    def test_fixed_reference_set_matches_original_vectorized_call(self):
        fixed_indices = torch.tensor([0, 2, 4, 6])
        selected = fixed_indices.repeat(self.query_deep.shape[0], 1)
        candidate_kernel = precompute_candidate_kernel_matrix(
            self.candidate_deep,
            self.candidate_raw,
            self.sigma,
            self.sigma0,
            self.epsilon,
            block_size=2,
        )
        fast = fast_adaptive_mmd_scores(
            self.candidate_deep,
            self.candidate_raw,
            self.query_deep,
            self.query_raw,
            selected,
            candidate_kernel,
            self.sigma,
            self.sigma0,
            self.epsilon,
        )

        reference_deep = self.candidate_deep.index_select(0, fixed_indices)
        reference_raw = self.candidate_raw.index_select(0, fixed_indices).flatten(1)
        expected = MMD_batch2(
            torch.cat((reference_deep, self.query_deep)),
            reference_deep.shape[0],
            torch.cat((reference_raw, self.query_raw.flatten(1))),
            self.sigma,
            self.sigma0,
            self.epsilon,
        )
        torch.testing.assert_close(fast, expected, rtol=1e-12, atol=1e-12)

    def test_kernel_precomputation_is_independent_of_block_size(self):
        single_block = precompute_candidate_kernel_matrix(
            self.candidate_deep,
            self.candidate_raw,
            self.sigma,
            self.sigma0,
            self.epsilon,
            block_size=self.candidate_deep.shape[0],
        )
        row_blocks = precompute_candidate_kernel_matrix(
            self.candidate_deep,
            self.candidate_raw,
            self.sigma,
            self.sigma0,
            self.epsilon,
            block_size=1,
        )
        torch.testing.assert_close(row_blocks, single_block, rtol=1e-12, atol=1e-12)

    def test_duplicate_reference_indices_are_rejected(self):
        invalid = self.indices.clone()
        invalid[1] = torch.tensor([1, 1, 3, 5])
        with self.assertRaisesRegex(ValueError, "duplicates for query 1"):
            slow_adaptive_mmd_scores(
                self.candidate_deep,
                self.candidate_raw,
                self.query_deep,
                self.query_raw,
                invalid,
                self.sigma,
                self.sigma0,
                self.epsilon,
            )

    def test_out_of_range_reference_indices_are_rejected(self):
        invalid = self.indices.clone()
        invalid[0, 0] = self.candidate_deep.shape[0]
        with self.assertRaisesRegex(IndexError, "out-of-range"):
            slow_adaptive_mmd_scores(
                self.candidate_deep,
                self.candidate_raw,
                self.query_deep,
                self.query_raw,
                invalid,
                self.sigma,
                self.sigma0,
                self.epsilon,
            )

    def test_invalid_kernel_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sigma must be greater than zero"):
            precompute_candidate_kernel_matrix(
                self.candidate_deep,
                self.candidate_raw,
                sigma=0,
                sigma0=self.sigma0,
                epsilon=self.epsilon,
            )
        with self.assertRaisesRegex(ValueError, "epsilon must be"):
            precompute_candidate_kernel_matrix(
                self.candidate_deep,
                self.candidate_raw,
                sigma=self.sigma,
                sigma0=self.sigma0,
                epsilon=1.1,
            )


if __name__ == "__main__":
    unittest.main()
