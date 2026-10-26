"""Boundary and preservation tests: python -m unittest discover -s src -p test_preprocess.py"""
from datetime import date
import unittest

import numpy as np
import pandas as pd

from preprocess import clean


def rows(n):
    return pd.DataFrame({"Occupancy": [1.] * n,
                         "Datetime": [f"2022-01-01 10:{i:02d}:00" for i in range(n)],
                         "Aula": ["A1"] * n, "Measure_ID": ["20220101_school_A1"] * n,
                         "CO2D": [500.] * n, "TairD": [20.] * n, "RHD": [50.] * n,
                         "TairC1": [20000.] * n})


class CleaningTests(unittest.TestCase):
    def run_clean(self, df):
        return clean(df, date(2026, 10, 26))

    def test_label_boundaries_and_invalid_values(self):
        df = rows(13)
        df["Occupancy"] = [0, 1, 5, 6, 15, 16, 60, -1, 61, 1.5, np.inf, np.nan, "bad"]
        out, decisions, steps, *_ = self.run_clean(df)
        self.assertEqual(out.occupancy_class.tolist(), [0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(len(out) + steps.removed_this_step.sum(), len(df))
        self.assertEqual(decisions.invalid_label.sum(), 5)
        self.assertEqual(decisions.missing_label.sum(), 1)

    def test_time_mismatch_retained_but_future_and_invalid_excluded(self):
        df = rows(4)
        df["Datetime"] = ["2022-01-02 10:00:00", "2222-01-01 10:00:00", "bad", "2026-10-26 23:59:59"]
        out, decisions, *_ = self.run_clean(df)
        self.assertEqual(out.source_row.tolist(), [2, 5])
        self.assertTrue(out.review_reasons.str.contains("date_measure_mismatch").all())
        self.assertEqual(decisions.exclusion_reason.tolist(), ["", "future_time", "invalid_time", ""])

    def test_sensor_bounds_and_overlapping_reasons(self):
        df = rows(6)
        df["CO2D"] = [250, 10000, 249, np.nan, np.inf, 500]
        df.loc[5, "RHD"] = 101
        df.loc[3, "Occupancy"] = np.nan
        out, decisions, steps, *_ = self.run_clean(df)
        self.assertEqual(out.source_row.tolist(), [2, 3])
        self.assertTrue(decisions.loc[3, "missing_D_core"])
        self.assertEqual(decisions.loc[3, "exclusion_reason"], "missing_label")
        self.assertEqual(len(out) + steps.removed_this_step.sum(), len(df))

    def test_duplicates_preserved_no_scale_repair_and_no_input_mutation(self):
        df = pd.concat([rows(1)] * 3, ignore_index=True)
        df.loc[1:, "Measure_ID"] = ["20220101_alias_A1", "20220101_other_A1"]
        df.loc[2, "Occupancy"] = 20  # Conflicting label still flagged/linked by core+time.
        original = df.copy(deep=True)
        out, _, _, _, edges, _ = self.run_clean(df)
        pd.testing.assert_frame_equal(original, df)
        self.assertEqual(len(out), 3)
        self.assertEqual(out.candidate_leakage_group.nunique(), 1)
        self.assertTrue(out.review_required.all())
        self.assertTrue((edges.evidence == "same_time_D_core").any())
        self.assertNotIn("TairC1", out)
        self.assertTrue(df.TairC1.eq(20000).all())


if __name__ == "__main__":
    unittest.main()
