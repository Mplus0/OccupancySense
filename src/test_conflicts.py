"""Conflict provenance review must stay read-only and consistent with upstream evidence."""
import hashlib
import json
import subprocess
import sys
import unittest

import pandas as pd

from analyze_conflicts import ROOT, OUT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConflictReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protected = [ROOT / "data/raw/ieq-dataset.csv", ROOT / "data/processed/A_C2_CO2_v1.csv"]
        cls.before = {str(p): digest(p) for p in cls.protected}
        cls.result = subprocess.run([sys.executable, str(ROOT / "src/analyze_conflicts.py")], capture_output=True)

    def test_open_issue_exit_code(self):
        self.assertEqual(self.result.returncode, 2, self.result.stderr.decode(errors="replace"))
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "OPEN")
        self.assertFalse(manifest["labels_modified"])
        self.assertFalse(manifest["training_performed"])

    def test_inputs_untouched(self):
        after = {str(p): digest(p) for p in self.protected}
        self.assertEqual(self.before, after)

    def test_every_conflict_key_attributed(self):
        attr = pd.read_csv(OUT / "conflict_key_attribution.csv")
        evidence = pd.read_csv(ROOT / "results/metrics/pretraining_audit/conflict_row_evidence.csv")
        kept = evidence[evidence.retained_in_C2.astype(bool)]
        self.assertEqual(attr.conflict_id.nunique(), 184)
        self.assertEqual(int(attr.retained_rows.sum()), len(kept))
        self.assertEqual(set(attr.conflict_id), set(kept.conflict_id))
        self.assertTrue(attr.identical_three_channel_text.all())
        self.assertEqual(set(attr.sensor_attribution), {"UNATTRIBUTABLE"})

    def test_conflicts_stay_in_train_split(self):
        attr = pd.read_csv(OUT / "conflict_key_attribution.csv")
        self.assertEqual(set(attr.proposed_split), {"train"})

    def test_quarantine_is_a_scenario_only(self):
        scenario = pd.read_csv(OUT / "quarantine_scenario.csv").set_index("scenario")
        single = pd.read_csv(ROOT / "data/processed/A_C2_CO2_v1.csv")
        self.assertEqual(int(scenario.loc["current_A_C2_CO2_v1", "rows"]), len(single))
        self.assertEqual(int(scenario.loc["quarantine_conflicted_session", "rows_removed"]),
                         int(scenario.loc["current_A_C2_CO2_v1", "rows"]) -
                         int(scenario.loc["quarantine_conflicted_session", "rows"]))
        self.assertLess(scenario.loc["quarantine_conflicted_session", "rows"],
                        scenario.loc["current_A_C2_CO2_v1", "rows"])

    def test_open_issues_recorded(self):
        issues = pd.read_csv(OUT / "open_issues.csv")
        self.assertIn("which_occupancy_label_belongs_to_the_observation", set(issues.issue))
        self.assertTrue(issues.loc[issues.issue.eq("which_occupancy_label_belongs_to_the_observation"), "resolved"].eq(False).all())


if __name__ == "__main__":
    unittest.main()
