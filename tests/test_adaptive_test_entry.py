from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from test_dMMD import (
    _reference_selection_mode,
    test_adaptive_dMMD,
    test_dMMD as run_legacy_dMMD,
)
from utils.adaptive_evaluation import AdaptiveReferencePool
from utils.adaptive_mmd import precompute_candidate_kernel_matrix


class _QueryDataset(Dataset):
    def __init__(self, features, label, prefix):
        self.features = features
        self.label = label
        self.prefix = prefix

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return (
            self.features[index],
            torch.tensor([self.label], dtype=torch.float32),
            f"{self.prefix}_{index}",
        )


class _FlattenNet(torch.nn.Module):
    def forward(self, inputs, out_feature=False):
        features = inputs.flatten(1)
        return inputs, features


class _EvaluationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.net = _FlattenNet()
        self.register_buffer("sigma", torch.tensor(2.0))
        self.register_buffer("sigma0_u", torch.tensor(0.5))
        self.register_buffer("ep", torch.tensor(0.1))
        self.is_smooth = True


class AdaptiveTestEntryTest(unittest.TestCase):
    def setUp(self):
        candidate_raw = torch.tensor(
            [
                [[0.0, 0.0]],
                [[0.2, 0.1]],
                [[0.7, 0.8]],
                [[1.0, 1.0]],
            ]
        )
        candidate_deep = candidate_raw.flatten(1)
        kernel = precompute_candidate_kernel_matrix(
            candidate_deep,
            candidate_raw,
            sigma=2.0,
            sigma0=0.5,
            epsilon=0.1,
            block_size=2,
        )
        self.pool = AdaptiveReferencePool(
            video_ids=("candidate_0", "candidate_1", "candidate_2", "candidate_3"),
            raw_features=candidate_raw,
            deep_features=candidate_deep,
            kernel_matrix=kernel,
        )
        self.model = _EvaluationModel()
        real = torch.tensor([[[0.1, 0.0]], [[0.3, 0.2]]])
        fake = torch.tensor([[[1.5, 1.5]], [[2.0, 2.0]]])
        self.loaders = {
            "real": DataLoader(_QueryDataset(real, 0, "real"), batch_size=2),
            "fake": DataLoader(_QueryDataset(fake, 1, "fake"), batch_size=2),
        }
        self.selection_cfg = SimpleNamespace(top_n=2, random_seed=42)

    def test_random_mode_produces_metrics_and_auditable_rows(self):
        metrics, query_rows, selection_rows = test_adaptive_dMMD(
            self.model,
            self.loaders,
            self.pool,
            "random_topn",
            self.selection_cfg,
            "SyntheticGenerator",
            "/tmp/checkpoint.pth",
            "abc123",
        )

        self.assertEqual(set(metrics), {"precision", "recall", "f1", "accuracy", "auroc"})
        self.assertEqual(len(query_rows), 4)
        self.assertEqual(len(selection_rows), 4 * self.selection_cfg.top_n)
        self.assertEqual({row["query_label"] for row in query_rows}, {0, 1})
        self.assertTrue(all(row["reference_mode"] == "random_topn" for row in query_rows))
        self.assertTrue(all(row["checkpoint_hash"] == "abc123" for row in query_rows))

    def test_fixed_mode_uses_one_shared_reference_set(self):
        metrics, query_rows, selection_rows = test_adaptive_dMMD(
            self.model,
            self.loaders,
            self.pool,
            "fixed_topn",
            self.selection_cfg,
            "SyntheticGenerator",
            "/tmp/checkpoint.pth",
            "abc123",
        )

        self.assertEqual(set(metrics), {"precision", "recall", "f1", "accuracy", "auroc"})
        self.assertEqual(len(query_rows), 4)
        selections_by_query = {}
        for row in selection_rows:
            key = (row["query_id"], row["query_label"])
            selections_by_query.setdefault(key, []).append(row["reference_id"])
        self.assertEqual(len(selections_by_query), 4)
        self.assertEqual(
            {tuple(reference_ids) for reference_ids in selections_by_query.values()},
            {("candidate_0", "candidate_1")},
        )
        self.assertEqual(
            {row["candidate_index"] for row in selection_rows},
            {0, 1},
        )

    def test_original_mode_can_return_id_aligned_query_rows(self):
        legacy_metrics = run_legacy_dMMD(
            self.model,
            self.loaders,
            self.pool.deep_features[:2],
            self.pool.raw_features[:2],
        )
        metrics, query_rows = run_legacy_dMMD(
            self.model,
            self.loaders,
            self.pool.deep_features[:2],
            self.pool.raw_features[:2],
            return_query_rows=True,
            generator_name="SyntheticGenerator",
            checkpoint_path="/tmp/checkpoint.pth",
            checkpoint_hash="abc123",
        )

        self.assertEqual(metrics, legacy_metrics)
        self.assertEqual(set(metrics), {"precision", "recall", "f1", "accuracy", "auroc"})
        self.assertEqual(len(query_rows), 4)
        self.assertEqual(
            {(row["query_id"], row["query_label"]) for row in query_rows},
            {("real_0", 0), ("real_1", 0), ("fake_0", 1), ("fake_1", 1)},
        )
        self.assertTrue(all(row["reference_mode"] == "original" for row in query_rows))
        self.assertTrue(all(row["top_n"] == 2 for row in query_rows))
        self.assertTrue(all(row["checkpoint_hash"] == "abc123" for row in query_rows))

    def test_missing_reference_configuration_falls_back_to_original(self):
        self.assertEqual(_reference_selection_mode({}), "original")


if __name__ == "__main__":
    unittest.main()
