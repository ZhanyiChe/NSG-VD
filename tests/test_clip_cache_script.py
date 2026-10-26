from pathlib import Path
import tempfile
import unittest

from scripts.cache_clip_video_features import _select_video_ids


class ClipCacheScriptTest(unittest.TestCase):
    def test_filters_mp4_before_start_and_limit(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            split_file = Path(temporary_dir) / "test_ids.txt"
            split_file.write_text(
                "A.mp4\nignored.gif\nB.mp4\nother.gif\nC.mp4\nD.mp4\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _select_video_ids(split_file, start_idx=1, limit=2),
                ["B.mp4", "C.mp4"],
            )

    def test_rejects_invalid_slice(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            split_file = Path(temporary_dir) / "test_ids.txt"
            split_file.write_text("A.mp4\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-negative"):
                _select_video_ids(split_file, start_idx=-1, limit=1)
            with self.assertRaisesRegex(ValueError, "positive integer or null"):
                _select_video_ids(split_file, start_idx=0, limit=0)


if __name__ == "__main__":
    unittest.main()
