import unittest

import pandas as pd

from utils.threshold_scan import (
    aggregate_real_only_runs,
    aggregate_run_macros,
    build_thresholds,
    macro_average_scan,
    scan_real_only_query_scores,
    scan_query_scores,
)


class ThresholdScanTest(unittest.TestCase):
    def setUp(self):
        self.query_scores = pd.DataFrame({
            "generator": ["A"] * 4 + ["B"] * 4,
            "query_label": [0, 0, 1, 1] * 2,
            "mmd_score": [0.2, 0.8, 1.2, 1.4, 0.6, 1.1, 0.9, 1.3],
            "decision_score": [0.2, 0.8, 1.2, 1.4, 0.6, 1.1, 0.9, 1.3],
        })

    def test_threshold_grid_is_inclusive_and_decimal_stable(self):
        self.assertEqual(
            build_thresholds(0.4, 0.7, 0.1),
            [0.4, 0.5, 0.6, 0.7],
        )

    def test_scan_uses_strict_greater_than_and_table1_macro_average(self):
        scan = scan_query_scores(self.query_scores, [1.0])
        macro = macro_average_scan(scan).iloc[0]

        self.assertAlmostEqual(macro["recall"], 0.75)
        self.assertAlmostEqual(macro["accuracy"], 0.75)
        self.assertAlmostEqual(macro["f1"], 0.75)
        self.assertAlmostEqual(macro["precision"], 0.75)
        self.assertAlmostEqual(macro["tnr"], 0.75)
        self.assertAlmostEqual(macro["fpr"], 0.25)
        self.assertEqual(macro["num_generators"], 2)

    def test_scan_can_use_explicit_decision_score_column(self):
        mmd_scan = scan_query_scores(self.query_scores, [1.0])
        decision_scan = scan_query_scores(
            self.query_scores,
            [1.0],
            score_column="decision_score",
        )

        self.assertEqual(set(decision_scan["score_column"]), {"decision_score"})
        pd.testing.assert_frame_equal(
            mmd_scan.drop(columns="score_column"),
            decision_scan.drop(columns="score_column"),
        )

    def test_scan_requires_the_selected_score_column(self):
        with self.assertRaisesRegex(ValueError, "decision_score"):
            scan_query_scores(
                self.query_scores.drop(columns="decision_score"),
                [1.0],
                score_column="decision_score",
            )

    def test_repeated_run_summary_uses_sample_standard_deviation(self):
        first = macro_average_scan(scan_query_scores(self.query_scores, [1.0]))
        second_scores = self.query_scores.copy()
        second_scores["mmd_score"] = [0.2, 0.8, 1.2, 1.4] * 2
        second = macro_average_scan(scan_query_scores(second_scores, [1.0]))
        for run_index, frame in enumerate((first, second)):
            frame.insert(0, "run_index", run_index)

        summary = aggregate_run_macros([first, second]).iloc[0]
        self.assertAlmostEqual(summary["accuracy_mean"], 0.875)
        self.assertAlmostEqual(summary["accuracy_std"], 0.1767766953)
        self.assertEqual(summary["num_runs"], 2)

    def test_rejects_generator_without_both_classes(self):
        invalid = self.query_scores[self.query_scores["query_label"] == 0]
        with self.assertRaisesRegex(ValueError, "both real and fake"):
            scan_query_scores(invalid, [1.0])

    def test_real_only_scan_requires_unique_real_queries(self):
        real_only = pd.DataFrame({
            "query_id": ["r0", "r1", "r2"],
            "query_label": [0, 0, 0],
            "mmd_score": [0.5, 1.0, 1.2],
        })
        scan = scan_real_only_query_scores(real_only, [1.0]).iloc[0]
        self.assertEqual(scan["num_real"], 3)
        self.assertAlmostEqual(scan["real_accuracy"], 2 / 3)
        self.assertAlmostEqual(scan["fpr"], 1 / 3)

        duplicated = pd.concat([real_only, real_only.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique query_id"):
            scan_real_only_query_scores(duplicated, [1.0])

    def test_real_only_scan_can_use_decision_score(self):
        real_only = pd.DataFrame({
            "query_id": ["r0", "r1", "r2"],
            "query_label": [0, 0, 0],
            "mmd_score": [0.1, 0.2, 0.3],
            "decision_score": [0.4, 0.5, 0.8],
        })

        scan = scan_real_only_query_scores(
            real_only,
            [0.5],
            score_column="decision_score",
        ).iloc[0]

        self.assertEqual(scan["score_column"], "decision_score")
        self.assertAlmostEqual(scan["real_accuracy"], 2 / 3)
        self.assertAlmostEqual(scan["mean_decision_score"], (0.4 + 0.5 + 0.8) / 3)
        self.assertTrue(pd.isna(scan["mean_mmd_score"]))

    def test_real_only_run_summary_uses_sample_standard_deviation(self):
        first = pd.DataFrame({
            "run_index": [0],
            "threshold": [1.0],
            "num_real": [500],
            "real_accuracy": [0.8],
            "fpr": [0.2],
        })
        second = pd.DataFrame({
            "run_index": [1],
            "threshold": [1.0],
            "num_real": [500],
            "real_accuracy": [0.9],
            "fpr": [0.1],
        })
        summary = aggregate_real_only_runs([first, second]).iloc[0]
        self.assertAlmostEqual(summary["real_accuracy_mean"], 0.85)
        self.assertAlmostEqual(summary["real_accuracy_std"], 0.0707106781)


if __name__ == "__main__":
    unittest.main()
