"""Verify saved group proposals independently; no dataset mutation."""
import hashlib
import json
import unittest

import pandas as pd

from analyze_split import ROOT, OUT


class ProposalTests(unittest.TestCase):
    def test_saved_proposals(self):
        data = pd.read_csv(ROOT / "data/processed/A_D_core_v1.csv")
        summaries = pd.read_csv(OUT / "candidate_summary.csv")
        for name in summaries.proposal.unique():
            groups = pd.read_csv(OUT / f"candidate_{name}_groups.csv")
            self.assertEqual(set(groups.Measure_ID), set(data.Measure_ID))
            self.assertTrue(groups.Measure_ID.is_unique)
            joined = data.merge(groups[["Measure_ID", "proposed_split", "school_candidate"]],
                                on="Measure_ID", validate="many_to_one")
            self.assertEqual(len(joined), len(data))
            for key in ["candidate_leakage_group", ["Datetime", "CO2D", "TairD", "RHD"]]:
                self.assertEqual(joined.groupby(key).proposed_split.nunique().max(), 1)
            if name.startswith("school_"):
                self.assertEqual(joined.groupby("school_candidate").proposed_split.nunique().max(), 1)
            parts = []
            for split in ["train", "validation", "test"]:
                part = joined[joined.proposed_split == split]
                report = summaries[(summaries.proposal == name) & (summaries.split == split)].iloc[0]
                self.assertEqual(len(part), report.rows)
                self.assertEqual(part.occupancy_class.nunique(), 4)
                self.assertEqual([int(part.occupancy_class.eq(c).sum()) for c in range(4)],
                                 [report[f"class_{c}"] for c in range(4)])
                parts.append(part)
            if name == "chronological":
                self.assertLess(parts[0].Datetime.max(), parts[1].Datetime.min())
                self.assertLess(parts[1].Datetime.max(), parts[2].Datetime.min())

    def test_input_hashes(self):
        manifest = json.loads((OUT / "analysis_manifest.json").read_text(encoding="utf-8"))
        for path, expected in manifest["input_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)
        audit = json.loads((ROOT / "results/metrics/audit_summary.json").read_text(encoding="utf-8"))
        for item in audit["datasets"].values():
            from pathlib import Path
            self.assertEqual(hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest(), item["sha256_before"])


if __name__ == "__main__":
    unittest.main()
