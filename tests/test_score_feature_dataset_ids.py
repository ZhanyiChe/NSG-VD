import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.feature_dataset.score_feature_dataset import ScoreFeaturesDataset
from utils.data_utils import get_data_loaders_for_mmd


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


if __name__ == "__main__":
    unittest.main()
