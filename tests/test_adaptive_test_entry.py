from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from test_dMMD import (
    _cache_query_clip_features,
    _collect_query_clip_cache_requests,
    _evaluation_mode,
    _merge_clip_cache_requests,
    _real_only_metrics,
    _reference_selection_mode,
    _score_original_real_query_loader,
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

    def test_missing_evaluation_configuration_preserves_standard_mode(self):
        self.assertEqual(_evaluation_mode({}), "standard")

    def test_clip_cache_requests_merge_repeated_real_queries(self):
        requests = _merge_clip_cache_requests([
            ("real", "MSR-VTT", "test", ("Video1", "video2")),
            ("fake", "GenA", "test", ("fake1",)),
            ("real", "MSR-VTT", "test", ("video1", "video3")),
        ])

        self.assertEqual(requests, [
            ("real", "MSR-VTT", "test", ("Video1", "video2", "video3")),
            ("fake", "GenA", "test", ("fake1",)),
        ])

    def test_real_only_clip_cache_request_uses_effective_dataset_ids(self):
        loader = DataLoader(_QueryDataset(
            torch.zeros(2, 1, 2), 0, "real"
        ), batch_size=2)
        loader.dataset.video_ids = ["kept_a", "kept_b"]
        cfg = OmegaConf.create({
            "real_only": {"real_model": "Kinetics-400", "split": "test"},
        })

        requests = _collect_query_clip_cache_requests(
            "real_only", cfg, real_only_loader=loader
        )

        self.assertEqual(requests, [
            ("real", "Kinetics-400", "test", ("kept_a", "kept_b")),
        ])

    def test_query_clip_precache_extracts_missing_then_skips_existing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "GenVideo"
            split_file = data_root / "split/real/MSR-VTT/test_ids.txt"
            split_file.parent.mkdir(parents=True)
            split_file.write_text("Video_A.mp4\nvideo_b.mp4\n", encoding="utf-8")
            for video_id in ("Video_A", "video_b"):
                frame_dir = data_root / f"video_frames/real/MSR-VTT/test/{video_id}"
                frame_dir.mkdir(parents=True)
                for frame_index in range(8):
                    (frame_dir / f"{frame_index:03d}.jpg").touch()

            cfg = OmegaConf.create({
                "data": {
                    "data_path": str(data_root),
                    "dataset_name": "GenVideo",
                },
                "reference_selection": {
                    "feature_cache_dir": str(data_root / "clip_features"),
                    "clip_model_name": "Fake-CLIP",
                    "clip_pretrained": "fake-weights",
                    "clip_num_frames": 8,
                    "clip_batch_size": 4,
                    "auto_cache_query_features": True,
                },
            })
            fake_model = SimpleNamespace(
                visual=SimpleNamespace(output_dim=3),
            )
            requests = [
                ("real", "MSR-VTT", "test", ("Video_A", "video_b")),
            ]

            with patch(
                    "test_dMMD.load_open_clip_model",
                    return_value=(fake_model, object(), "test-version")) as load_mock, patch(
                    "test_dMMD.encode_video_frames",
                    return_value=torch.tensor([1.0, 0.0, 0.0])) as encode_mock:
                _cache_query_clip_features(cfg, requests, torch.device("cpu"))
                self.assertEqual(load_mock.call_count, 1)
                self.assertEqual(encode_mock.call_count, 2)

            with patch(
                    "test_dMMD.load_open_clip_model",
                    side_effect=AssertionError("OpenCLIP must not reload")) as load_mock, patch(
                    "test_dMMD.encode_video_frames",
                    side_effect=AssertionError("Cached videos must not be re-encoded")) as encode_mock:
                _cache_query_clip_features(cfg, requests, torch.device("cpu"))
                load_mock.assert_not_called()
                encode_mock.assert_not_called()

    def test_real_only_metrics_report_true_negatives_and_false_positives(self):
        metrics = _real_only_metrics(torch.tensor([0.2, 1.0, 1.1, 2.0]))

        self.assertEqual(metrics["num_real"], 4)
        self.assertEqual(metrics["true_negatives"], 2)
        self.assertEqual(metrics["false_positives"], 2)
        self.assertEqual(metrics["tnr"], 0.5)
        self.assertEqual(metrics["fpr"], 0.5)
        self.assertAlmostEqual(metrics["mean_mmd_score"], 1.075)

    def test_original_real_only_scorer_returns_one_row_per_real_query(self):
        real = torch.tensor([[[0.1, 0.0]], [[0.3, 0.2]]])
        loader = DataLoader(_QueryDataset(real, 0, "real"), batch_size=2)

        scores, rows, metrics = _score_original_real_query_loader(
            self.model,
            loader,
            self.pool.deep_features[:2],
            self.pool.raw_features[:2],
            "SyntheticReal",
            "/tmp/checkpoint.pth",
            "abc123",
        )

        self.assertEqual(len(scores), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(metrics["num_real"], 2)
        self.assertEqual({row["query_label"] for row in rows}, {0})
        self.assertEqual({row["query_id"] for row in rows}, {"real_0", "real_1"})
        self.assertTrue(all(row["reference_mode"] == "original" for row in rows))


if __name__ == "__main__":
    unittest.main()
