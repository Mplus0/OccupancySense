"""Repair evidence boundaries and saved C2 candidate invariants."""
import hashlib
import json
import unittest

import pandas as pd

from preprocess_c2 import ROOT, OUT, PROCESSED, supported_repairs


class RepairTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({"Datetime":["2022-01-01 10:00:00","2022-01-01 10:01:00","2022-01-01 10:11:00"],
                             "Measure_ID":["session"]*3,"TairC2":[20.,20000.,20000.],
                             "RHC2":[40.,40000.,40000.],"TairD":[20.]*3,"RHD":[40.]*3})

    def test_past_original_anchor_and_no_repair_chaining(self):
        data=self.fixture(); original=data.copy(deep=True)
        result,log=supported_repairs(data,True)
        self.assertEqual(result.TairC2.tolist(),[20.,20.,20000.])
        self.assertEqual(result.RHC2.tolist(),[40.,40.,40000.])
        self.assertEqual(log.action.eq("divide_1000_supported_candidate").sum(),2)
        pd.testing.assert_frame_equal(data,original)

    def test_requires_checksum_and_both_references(self):
        data=self.fixture()
        result,_=supported_repairs(data,False)
        self.assertEqual(result.TairC2.tolist(),data.TairC2.tolist())
        data.loc[1,"TairD"]=30.
        result,_=supported_repairs(data,True)
        self.assertEqual(result.loc[1,"TairC2"],20000.)
        self.assertEqual(result.loc[1,"RHC2"],40.)

    def test_no_future_or_cross_group_anchor(self):
        data=self.fixture()
        data.loc[0,"Datetime"]="2022-01-01 10:02:00"
        result,_=supported_repairs(data,True)
        self.assertEqual(result.loc[1,"TairC2"],20000.)
        data=self.fixture(); data.loc[0,"Measure_ID"]="different"
        result,_=supported_repairs(data,True)
        self.assertEqual(result.loc[1,"TairC2"],20000.)


class ArtifactTests(unittest.TestCase):
    def test_source_preservation_and_exact_repair_logs(self):
        manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
        for path,expected in manifest["protected_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected)
        raw=pd.read_csv(ROOT/"data/raw/ieq-dataset.csv")
        single=pd.read_csv(PROCESSED/"A_C2_CO2_v1.csv")
        self.assertEqual(len(single),8139)
        self.assertEqual(single.occupancy_class.value_counts().sort_index().tolist(),[545,2174,2138,3282])
        self.assertTrue((raw.loc[single.source_row-2,"CO2C2"].to_numpy()==single.CO2C2.to_numpy()).all())
        multi=pd.read_csv(PROCESSED/"A_C2_core_supported_candidate_v1.csv")
        log=pd.read_csv(OUT/"repair_decisions.csv")
        applied=log[log.action.eq("divide_1000_supported_candidate")].set_index(["source_row","field"])
        for col in ["TairC2","RHC2"]:
            changed=multi[multi[col+"_repaired"]]
            for row in changed.itertuples():
                evidence=applied.loc[(row.source_row,col)]
                self.assertEqual(getattr(row,col),evidence.original_value/1000)
                self.assertTrue(evidence.past_anchor_valid and evidence.reference_valid)
                self.assertEqual(str(evidence.admissible_divisors),"1000.0" if isinstance(evidence.admissible_divisors,float) else "1000")

    def test_proposed_splits_are_grouped_and_covered(self):
        data=pd.read_csv(PROCESSED/"A_C2_CO2_v1.csv")
        for level in ["school_unit","candidate_session"]:
            groups=pd.read_csv(OUT/f"candidate_A_C2_CO2_v1_{level}_groups.csv")
            self.assertEqual(set(groups.Measure_ID),set(data.Measure_ID))
            proposed=data.merge(groups[["Measure_ID","proposed_split"]],on="Measure_ID",validate="many_to_one")
            self.assertEqual(len(proposed),len(data))
            self.assertEqual(proposed.groupby("candidate_session").proposed_split.nunique().max(),1)
            self.assertEqual(proposed.groupby(["Datetime","CO2C2"]).proposed_split.nunique().max(),1)
            if level=="school_unit":
                self.assertEqual(proposed.groupby("school_candidate").proposed_split.nunique().max(),1)
            for _,part in proposed.groupby("proposed_split"):
                self.assertEqual(part.occupancy_class.nunique(),4)
                self.assertGreaterEqual(part.occupancy_class.value_counts().min(),20)
                self.assertGreaterEqual(part.groupby("occupancy_class").candidate_session.nunique().min(),2)


if __name__=="__main__":
    unittest.main()
