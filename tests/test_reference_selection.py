import random
import unittest

import torch

from utils.reference_selection import (
    normalize_video_id,
    select_clip_bottomn,
    select_clip_topn,
    select_fixed_topn,
    select_random_topn,
    validate_reference_selection_mode,
)


class ReferenceSelectionTest(unittest.TestCase):
    def test_validate_mode_normalizes_case_and_whitespace(self):
        self.assertEqual(validate_reference_selection_mode(" CLIP_TOPN "), "clip_topn")
        self.assertEqual(validate_reference_selection_mode(" FIXED_TOPN "), "fixed_topn")
        self.assertEqual(
            validate_reference_selection_mode(" CLIP_BOTTOMN "),
            "clip_bottomn",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported reference-selection mode"):
            validate_reference_selection_mode("nearest")

    def test_normalize_video_id_handles_paths_extensions_and_case(self):
        self.assertEqual(normalize_video_id(" /data/MSRVTT_42.MP4 "), "msrvtt_42")
        self.assertEqual(normalize_video_id(r"C:\\videos\\MSRVTT_42.mp4"), "msrvtt_42")

    def test_clip_topn_sorts_by_similarity_and_excludes_query(self):
        candidate_ids = ["A.mp4", "query.mp4", "C.mp4", "B.mp4"]
        candidate_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.8, 0.6],
                [0.8, 0.6],
            ]
        )
        query_embedding = torch.tensor([1.0, 0.0])

        selections = select_clip_topn(
            candidate_ids,
            candidate_embeddings,
            "/queries/QUERY.MP4",
            query_embedding,
            top_n=3,
        )

        self.assertEqual([item.reference_id for item in selections], ["A.mp4", "B.mp4", "C.mp4"])
        self.assertEqual([item.candidate_index for item in selections], [0, 3, 2])
        self.assertEqual([item.rank for item in selections], [1, 2, 3])
        self.assertAlmostEqual(selections[0].similarity, 1.0)

    def test_clip_topn_rejects_unnormalized_embeddings(self):
        with self.assertRaisesRegex(ValueError, "L2-normalized"):
            select_clip_topn(
                ["a.mp4", "b.mp4"],
                torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
                "query.mp4",
                torch.tensor([1.0, 0.0]),
                top_n=1,
            )

    def test_clip_bottomn_sorts_ascending_and_excludes_query(self):
        candidate_ids = ["A.mp4", "query.mp4", "C.mp4", "B.mp4", "D.mp4"]
        candidate_embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 0.0],
            ]
        )
        query_embedding = torch.tensor([1.0, 0.0])

        selections = select_clip_bottomn(
            candidate_ids,
            candidate_embeddings,
            "/queries/QUERY.MP4",
            query_embedding,
            top_n=3,
        )

        self.assertEqual(
            [item.reference_id for item in selections],
            ["D.mp4", "B.mp4", "C.mp4"],
        )
        self.assertEqual([item.candidate_index for item in selections], [4, 3, 2])
        self.assertEqual([item.rank for item in selections], [1, 2, 3])
        self.assertAlmostEqual(selections[0].similarity, -1.0)

    def test_clip_topn_and_bottomn_are_disjoint_when_pool_is_large_enough(self):
        candidate_ids = [f"candidate_{index}.mp4" for index in range(6)]
        angles = torch.linspace(0, torch.pi, steps=6)
        candidate_embeddings = torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=1
        )
        query_embedding = torch.tensor([1.0, 0.0])

        top = select_clip_topn(
            candidate_ids, candidate_embeddings, "query.mp4", query_embedding, 2
        )
        bottom = select_clip_bottomn(
            candidate_ids, candidate_embeddings, "query.mp4", query_embedding, 2
        )

        self.assertTrue(
            {item.candidate_index for item in top}.isdisjoint(
                item.candidate_index for item in bottom
            )
        )

    def test_fixed_topn_always_selects_the_first_candidates(self):
        candidate_ids = ["query.mp4", "B.mp4", "C.mp4", "D.mp4"]

        selections = select_fixed_topn(candidate_ids, top_n=3)

        self.assertEqual(
            [item.reference_id for item in selections],
            ["query.mp4", "B.mp4", "C.mp4"],
        )
        self.assertEqual([item.candidate_index for item in selections], [0, 1, 2])
        self.assertEqual([item.rank for item in selections], [1, 2, 3])

    def test_random_topn_is_reproducible_and_does_not_touch_global_rng(self):
        candidate_ids = [f"candidate_{index}.mp4" for index in range(30)]
        query_id = "query.mp4"

        random.seed(1234)
        expected_next_value = random.random()
        random.seed(1234)

        first = select_random_topn(candidate_ids, query_id, top_n=8, random_seed=42)
        second = select_random_topn(candidate_ids, query_id, top_n=8, random_seed=42)

        self.assertEqual(first, second)
        self.assertEqual(random.random(), expected_next_value)
        self.assertEqual(len({item.reference_id for item in first}), 8)

    def test_random_topn_changes_with_seed_and_excludes_query(self):
        candidate_ids = ["query.mp4"] + [f"candidate_{index}.mp4" for index in range(30)]

        first = select_random_topn(candidate_ids, "QUERY.MP4", top_n=8, random_seed=1)
        second = select_random_topn(candidate_ids, "QUERY.MP4", top_n=8, random_seed=2)

        self.assertNotEqual(first, second)
        self.assertNotIn("query.mp4", [item.reference_id for item in first])

    def test_selection_rejects_insufficient_candidates_after_exclusion(self):
        with self.assertRaisesRegex(ValueError, "only 1 of 2 candidates remain"):
            select_random_topn(
                ["query.mp4", "other.mp4"],
                "query.mp4",
                top_n=2,
                random_seed=0,
            )

    def test_selection_rejects_duplicate_normalized_candidate_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate normalized IDs"):
            select_random_topn(
                ["/a/Video.MP4", "/b/video.mp4"],
                "query.mp4",
                top_n=1,
                random_seed=0,
            )


if __name__ == "__main__":
    unittest.main()
