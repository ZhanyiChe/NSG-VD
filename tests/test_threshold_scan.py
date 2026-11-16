import unittest

import pandas as pd

from utils.threshold_scan import (
    aggregate_run_macros,
    build_thresholds,
    macro_average_scan,
    scan_query_scores,
)


class ThresholdScanTest(unittest.TestCase):
    def setUp(self):
        self.query_scores = pd.DataFrame({
            "generator": ["A"] * 4 + ["B"] * 4,
            "query_label": [0, 0, 1, 1] * 2,
            "mmd_score": [0.2, 0.8, 1.2, 1.4, 0.6, 1.1, 0.9, 1.3],
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


if __name__ == "__main__":
    unittest.main()
