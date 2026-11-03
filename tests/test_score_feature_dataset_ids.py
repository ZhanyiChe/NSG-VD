import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from data.feature_dataset.score_feature_dataset import ScoreFeaturesDataset
from data.video_frames_dataset import VideoFramesDataset
from utils.data_utils import get_data_loaders_for_mmd, get_ref_dataloader


class _DataConfig:
    num_workers = 0


class _IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


class ScoreFeatureDatasetVideoIdTest(unittest.TestCase):
    def _dataset_without_initialization(self, input_dir, return_video_id):
        dataset = ScoreFeaturesDataset.__new__(ScoreFeaturesDataset)
        dataset.input_dir = str(input_dir)
        dataset.video_ids = ["video_001"]
        dataset.feature_type = "velocity"
        dataset.filter_frames = False
        dataset.label = "real"
        dataset.return_video_id = return_video_id
        return dataset

    def test_default_item_keeps_original_two_tuple(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            np.save(input_dir / "nsg_video_001.npy", np.ones((2, 1, 2, 2), dtype=np.float32))
            dataset = self._dataset_without_initialization(input_dir, False)

            item = dataset[0]

            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], torch.Tensor)
            np.testing.assert_array_equal(item[1], np.array([0], dtype=np.float32))

    def test_optional_item_returns_aligned_video_id(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            np.save(input_dir / "nsg_video_001.npy", np.ones((2, 1, 2, 2), dtype=np.float32))
            dataset = self._dataset_without_initialization(input_dir, True)

            feature, label, video_id = dataset[0]

            self.assertEqual(video_id, "video_001")
            self.assertEqual(tuple(feature.shape), (2, 1, 2, 2))
            np.testing.assert_array_equal(label, np.array([0], dtype=np.float32))

    def test_mmd_loader_can_disable_shuffle_without_changing_default(self):
        datasets = {"real": _IndexedDataset(20), "fake": _IndexedDataset(20)}

        ordered = get_data_loaders_for_mmd(
            _DataConfig(), datasets, batch_size=20, shuffle=False
        )

        self.assertEqual(next(iter(ordered["real"])).tolist(), list(range(20)))
        self.assertIsInstance(ordered["real"].sampler, torch.utils.data.SequentialSampler)
        default = get_data_loaders_for_mmd(_DataConfig(), datasets, batch_size=20)
        self.assertIsInstance(default["real"].sampler, torch.utils.data.RandomSampler)

    def test_reference_loader_forwards_custom_selection_options(self):
        data_cfg = SimpleNamespace(
            data_path="/data",
            dataset_name="GenVideo",
            diffuse_steps=5,
            feature_type="velocity",
            batch_size=24,
            num_workers=0,
        )
        dataset_sentinel = _IndexedDataset(1)
        with patch(
                "utils.data_utils.ScoreFeaturesDataset",
                return_value=dataset_sentinel) as dataset_class:
            loader = get_ref_dataloader(
                data_cfg,
                "MSR-VTT",
                mode="test",
                load_len=80,
                start_idx=20,
                ids_file="/tmp/custom_ids.txt",
                return_video_id=True,
            )

        kwargs = dataset_class.call_args.kwargs
        self.assertEqual(kwargs["mode"], "test")
        self.assertEqual(kwargs["load_len"], 80)
        self.assertEqual(kwargs["start_idx"], 20)
        self.assertEqual(kwargs["ids_file"], "/tmp/custom_ids.txt")
        self.assertTrue(kwargs["return_video_id"])
        self.assertIs(loader.dataset, dataset_sentinel)

    def test_custom_id_file_accepts_ids_with_or_without_mp4_suffix(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_root = Path(temporary_dir) / "GenVideo"
            ids_file = Path(temporary_dir) / "reference_ids.txt"
            ids_file.write_text(
                "Video_A\nvideo_b.mp4\n\n",
                encoding="utf-8",
            )
            for video_id in ("Video_A", "video_b"):
                frame_dir = (
                    data_root
                    / "video_frames/real/MSR-VTT/val"
                    / video_id
                )
                frame_dir.mkdir(parents=True)
                for frame_index in range(8):
                    (frame_dir / f"frame{frame_index}.jpg").touch()

            dataset = VideoFramesDataset(
                data_path=str(data_root),
                dataset_name="GenVideo",
                generation_model="MSR-VTT",
                mode="val",
                len_load=2,
                num_frames=8,
                ids_file=str(ids_file),
            )

        self.assertEqual(dataset.video_ids, ["Video_A", "video_b"])


if __name__ == "__main__":
    unittest.main()
