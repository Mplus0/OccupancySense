"""Independent checks on saved read-only zero-coverage analysis."""
import hashlib
import json
import unittest

import pandas as pd

from analyze_zero_coverage import ROOT, OUT


class ZeroCoverageTests(unittest.TestCase):
    def test_trace_and_loss(self):
        raw = pd.read_csv(ROOT / "data/raw/ieq-dataset.csv")
        trace = pd.read_csv(OUT / "zero_row_trace.csv")
        self.assertEqual(trace.source_row.tolist(), (raw.index[raw.Occupancy.eq(0)]+2).tolist())
        self.assertEqual(len(trace), 556)
        self.assertEqual(int(trace.D_core_retained.sum()), 192)
        excluded = trace[~trace.D_core_retained]
        self.assertEqual(len(excluded), 364)
        self.assertTrue(excluded[["CO2D","TairD","RHD"]].isna().all().all())
        self.assertEqual(set(excluded.exclusion_reason), {"missing_D_core"})
        for level in ["school", "room", "measure", "date", "school_date", "session"]:
            counts = pd.read_csv(OUT / f"zero_by_{level}.csv")
            self.assertEqual(int(counts.raw_zero.sum()), 556)
            self.assertEqual(int(counts.retained_zero.sum()), 192)
            self.assertEqual(int(counts.lost_zero.sum()), 364)

    def test_raw_sensor_coverage(self):
        raw = pd.read_csv(ROOT / "data/raw/ieq-dataset.csv")
        zero = raw.Occupancy.eq(0)
        c2 = raw.CO2C2.between(250,10000)
        core = c2 & raw.TairC2.between(-50,60) & raw.RHC2.between(0,100)
        self.assertEqual(int((zero & c2).sum()), 545)
        self.assertEqual(int((zero & core).sum()), 7)
        self.assertTrue(raw.loc[zero & ~c2,"CO2C2"].isna().all())
        scenarios = pd.read_csv(OUT / "sensor_coverage_scenarios.csv").set_index("scenario")
        self.assertEqual(scenarios.loc["C2_CO2_raw","zero_rows"], 545)
        self.assertEqual(scenarios.loc["C2_CO2_raw","recovered_excluded_zero"], 353)
        self.assertEqual(scenarios.loc["C2_core_raw","zero_rows"], 7)

    def test_protected_inputs_unchanged(self):
        manifest = json.loads((OUT / "analysis_manifest.json").read_text(encoding="utf-8"))
        for path, expected in manifest["input_hashes_unchanged"].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
