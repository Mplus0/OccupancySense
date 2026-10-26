"""The current source-label conflict must stop training; no models are fitted."""
import json
import subprocess
import sys
import unittest

import pandas as pd

from check_training_readiness import ROOT, OUT


class ReadinessTests(unittest.TestCase):
    def test_blocking_exit_code(self):
        result=subprocess.run([sys.executable,str(ROOT/"src/check_training_readiness.py")],
                              capture_output=True)
        self.assertEqual(result.returncode,2,result.stderr.decode(errors="replace"))
        manifest=json.loads((OUT/"audit_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"],"BLOCKED")
        self.assertEqual(manifest["blockers"],["unresolved_source_label_conflicts"])
        self.assertFalse(manifest["model_training_performed"])
        self.assertFalse(manifest["formal_split_created"])

    def test_conflict_pair_from_original_data(self):
        raw=pd.read_csv(ROOT/"data/raw/ieq-dataset.csv")
        a,b=raw.iloc[1277-2],raw.iloc[10482-2]
        for col in ["Datetime","CO2C2","TairC2","RHC2"]:
            self.assertEqual(a[col],b[col])
        self.assertEqual(a.Occupancy,12)
        self.assertEqual(b.Occupancy,23)
        evidence=pd.read_csv(OUT/"conflict_row_evidence.csv")
        kept=evidence[evidence.retained_in_C2]
        self.assertEqual(kept.source_row.nunique(),368)
        self.assertEqual(kept.conflict_id.nunique(),184)
        self.assertEqual(set(kept.proposed_split),{"train"})
        self.assertTrue(kept.groupby("conflict_id").occupancy_class.nunique().gt(1).all())

    def test_repairs_do_not_touch_targets_or_cross_splits(self):
        checks=pd.read_csv(OUT/"checks.csv").set_index("check")
        for key in ["CO2C2_matches_raw","Occupancy_matches_raw","four_class_mapping_matches_raw",
                    "repair_evidence_matches_raw","repair_references_stay_within_session",
                    "protected_data_and_models_unchanged"]:
            self.assertTrue(checks.loc[key,"passed"])
        repairs=pd.read_csv(OUT/"repair_evidence_checks.csv")
        self.assertEqual(len(repairs),3136)
        self.assertTrue(repairs.evidence_matches_raw.all())
        self.assertFalse(repairs.cross_split_reference.any())


if __name__=="__main__":
    unittest.main()
