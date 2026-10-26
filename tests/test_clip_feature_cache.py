from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from utils.clip_feature_cache import ClipFeatureCache, ClipFeatureCacheSpec
from utils.clip_video_features import encode_video_frames, resolve_video_frame_paths


class _MeanRgbEncoder(torch.nn.Module):
    def encode_image(self, images):
        return images.mean(dim=(2, 3))


def _image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _cache_spec():
    return ClipFeatureCacheSpec(
        model_name="test-model",
        pretrained="test-weights",
        open_clip_version="test-version",
        embedding_dim=3,
        num_frames=2,
        frame_source="test-frames",
        preprocess_fingerprint="abc123",
        split_sha256="def456",
        dataset_name="GenVideo",
        label="real",
        generation_model="MSR-VTT",
        split="val",
    )


class ClipFeatureCacheTest(unittest.TestCase):
    def test_video_aggregation_normalizes_frames_then_mean(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            frame_dir = Path(temporary_dir)
            Image.new("RGB", (4, 4), (255, 0, 0)).save(frame_dir / "frame1.jpg")
            Image.new("RGB", (4, 4), (0, 255, 0)).save(frame_dir / "frame2.jpg")
            frame_paths = resolve_video_frame_paths(frame_dir, num_frames=2)

            embedding = encode_video_frames(
                _MeanRgbEncoder(),
                _image_to_tensor,
                frame_paths,
                torch.device("cpu"),
                frame_batch_size=1,
            )

            expected = torch.tensor([2 ** -0.5, 2 ** -0.5, 0.0])
            self.assertTrue(torch.allclose(embedding, expected, atol=2e-2, rtol=2e-2))
            self.assertAlmostEqual(torch.linalg.vector_norm(embedding).item(), 1.0, places=6)

    def test_resolve_frames_requires_exact_count(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            frame_dir = Path(temporary_dir)
            Image.new("RGB", (2, 2), (0, 0, 0)).save(frame_dir / "frame1.jpg")
            with self.assertRaisesRegex(ValueError, "Expected exactly 2"):
                resolve_video_frame_paths(frame_dir, num_frames=2)

    def test_cache_round_trip_and_incremental_missing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache = ClipFeatureCache(Path(temporary_dir), _cache_spec())
            embedding = torch.tensor([1.0, 0.0, 0.0])

            self.assertEqual(cache.missing(["A.mp4", "B.mp4"]), ("A.mp4", "B.mp4"))
            cache.put(
                "A.mp4",
                embedding,
                relative_frame_dir="real/MSR-VTT/val/A",
                frame_names=["frame1.jpg", "frame2.jpg"],
            )

            self.assertTrue(cache.contains("/different/path/a.MP4"))
            self.assertEqual(cache.missing(["A.mp4", "B.mp4"]), ("B.mp4",))
            self.assertTrue(torch.equal(cache.load("a.mp4"), embedding))

            reloaded = ClipFeatureCache(Path(temporary_dir), _cache_spec())
            self.assertTrue(torch.equal(reloaded.load_many(["A.mp4"]), embedding.unsqueeze(0)))

    def test_cache_rejects_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            ClipFeatureCache(cache_dir, _cache_spec())
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                ClipFeatureCache(
                    cache_dir,
                    replace(_cache_spec(), pretrained="different-weights"),
                )

    def test_cache_rejects_unnormalized_embedding(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache = ClipFeatureCache(Path(temporary_dir), _cache_spec())
            with self.assertRaisesRegex(ValueError, "L2-normalized"):
                cache.put(
                    "A.mp4",
                    torch.tensor([2.0, 0.0, 0.0]),
                    relative_frame_dir="real/MSR-VTT/val/A",
                    frame_names=["frame1.jpg", "frame2.jpg"],
                )

    def test_missing_validates_existing_entries(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache = ClipFeatureCache(Path(temporary_dir), _cache_spec())
            cache.put(
                "A.mp4",
                torch.tensor([1.0, 0.0, 0.0]),
                relative_frame_dir="real/MSR-VTT/val/A",
                frame_names=["frame1.jpg", "frame2.jpg"],
            )
            embedding_path, _ = cache._paths("A.mp4")
            embedding_path.write_bytes(b"not-a-numpy-file")

            with self.assertRaisesRegex(ValueError, "Cannot read CLIP embedding"):
                cache.missing(["A.mp4"])


if __name__ == "__main__":
    unittest.main()
