"""Exclude unresolved conflict records, freeze the existing session plan, train baselines.

Only validation scores select the model. Run evaluate.py afterwards for the test.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import sha, write_csv, classes
from evaluate import metrics, save_metrics

ROOT=Path(__file__).resolve().parents[1]
SEED=42
FEATURES=["CO2C2"]


def prepare(directory, split_dir):
    raw_path=ROOT/"data/raw/ieq-dataset.csv"
    source_path=ROOT/"data/processed/A_C2_CO2_v1.csv"
    plan_path=ROOT/"results/metrics/c2_preprocessing/candidate_A_C2_CO2_v1_candidate_session_groups.csv"
    raw=pd.read_csv(raw_path); source=pd.read_csv(source_path); plan=pd.read_csv(plan_path)
    raw_classes=pd.Series(pd.NA,index=raw.index,dtype="Int64")
    valid=raw.Occupancy.notna()&np.isfinite(raw.Occupancy)&raw.Occupancy.between(0,60)&raw.Occupancy.mod(1).eq(0)
    raw_classes.loc[valid]=classes(raw.loc[valid,"Occupancy"]).astype("Int64")
    key=["Datetime","CO2C2","TairC2","RHC2"]
    conflict_map={}; key_count=0
    for values,block in raw.loc[raw[key].notna().all(axis=1)&raw.duplicated(key,keep=False)].groupby(key,sort=True):
        if raw_classes.loc[block.index].nunique()>1:
            key_count+=1
            conflict_map.update({int(i)+2:key_count for i in block.index})
    affected=source.source_row.isin(conflict_map)
    if key_count!=184 or int(affected.sum())!=368:
        raise RuntimeError("Conflict scope changed from reviewed 184 keys / 368 rows; review before training")
    if set(source.loc[affected,"source_row"])!=set(source.loc[source.class_conflict_review,"source_row"]):
        raise RuntimeError("Conflict reconstruction disagrees with prior review")
    excluded=source.loc[affected].copy()
    excluded["conflict_key_id"]=excluded.source_row.map(conflict_map)
    excluded["action"]="exclude_all_records_for_unresolved_conflicting_key"
    excluded["reason"]="same raw time+C2 three-channel observation has different classes; source cannot identify correct target; no relabeling"
    excluded["previous_split"]=excluded.Measure_ID.map(plan.set_index("Measure_ID").proposed_split)
    write_csv(excluded,directory/"excluded_conflicts.csv")
    cleaned=source.loc[~affected].copy()
    if not plan.Measure_ID.is_unique or set(plan.Measure_ID)!=set(source.Measure_ID):
        raise RuntimeError("Candidate plan must cover all measurement IDs exactly once")
    full=cleaned.merge(plan[["Measure_ID","proposed_split"]],on="Measure_ID",validate="many_to_one").rename(columns={"proposed_split":"split"})
    checks=[]
    def require(name,condition,detail):
        checks.append(dict(check=name,passed=bool(condition),detail=detail))
    aligned=raw.iloc[full.source_row.to_numpy()-2]
    require("source_rows_unique",full.source_row.is_unique,"No source row duplicated by joins")
    require("CO2C2_unchanged",np.array_equal(full.CO2C2.to_numpy(),aligned.CO2C2.to_numpy()),"Exact comparison with raw source")
    require("Occupancy_unchanged",np.array_equal(full.Occupancy.to_numpy(),aligned.Occupancy.to_numpy()),"No label guessing or rewriting")
    require("class_mapping_correct",np.array_equal(full.occupancy_class.to_numpy(),classes(aligned.Occupancy).to_numpy()),"Original four class boundaries")
    require("all_conflicts_excluded",not full.source_row.isin(conflict_map).any(),"All retained records associated with 184 ambiguous keys excluded")
    require("CO2_finite",np.isfinite(full.CO2C2).all(),"No imputation needed")
    for col in ["Measure_ID","candidate_session","candidate_leakage_group",["Datetime","CO2C2"]]:
        crossings=int(full.groupby(col,dropna=False).split.nunique().gt(1).sum())
        require("no_cross_split_"+str(col),crossings==0,f"crossing keys={crossings}")
    links=pd.read_csv(ROOT/"results/metrics/c2_preprocessing/source_links.csv")
    mapping=plan.set_index("Measure_ID").proposed_split.to_dict()
    crossings=sum(mapping[r.left]!=mapping[r.right] for r in links.itertuples() if r.left in mapping and r.right in mapping)
    require("known_raw_source_links_not_crossed",crossings==0,f"crossing links={crossings}")
    require("valid_split_names",set(full.split)=={"train","validation","test"},"Frozen original candidate plan")
    summaries=[]
    for split in ["train","validation","test"]:
        part=full[full.split.eq(split)]
        require(split+"_all_classes",part.occupancy_class.nunique()==4,"Coverage audit only; no split search")
        row=dict(split=split,rows=len(part),fraction=len(part)/len(full),measure_groups=part.Measure_ID.nunique(),candidate_sessions=part.candidate_session.nunique())
        for c in range(4):
            subset=part[part.occupancy_class.eq(c)]
            row[f"class_{c}"]=len(subset); row[f"class_{c}_sessions"]=subset.candidate_session.nunique()
        summaries.append(row)
    write_csv(pd.DataFrame(checks),directory/"leakage_checks.csv")
    if not all(r["passed"] for r in checks):
        raise RuntimeError("Data or leakage checks failed; see leakage_checks.csv")
    write_csv(pd.DataFrame(summaries),directory/"split_summary.csv")
    new_path=ROOT/"data/processed/A_C2_CO2_no_conflicts_v1.csv"
    if new_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(new_path),cleaned.reset_index(drop=True),check_dtype=False)
    else:
        write_csv(cleaned,new_path)
    for split in ["train","validation","test"]:
        write_csv(full[full.split.eq(split)],split_dir/f"{split}.csv")
    write_csv(full[["source_row","Measure_ID","candidate_session","split"]],directory/"split_assignments.csv")
    return {"original_dataset_sha256":sha(source_path),"raw_sha256":sha(raw_path),"candidate_plan_sha256":sha(plan_path),
            "cleaned_dataset_sha256":sha(new_path),"removed_records":int(affected.sum()),"removed_keys":key_count,
            "split_sha256":{s:sha(split_dir/f"{s}.csv") for s in ["train","validation","test"]}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="co2_baseline_v1")
    args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name or args.run_name in [".",".."]:
        parser.error("run-name must be a directory name")
    directory=ROOT/"results/metrics"/args.run_name
    if (directory/"selection.json").exists():
        parser.error("Completed run already exists; use a new --run-name for reproduction")
    model_dir=ROOT/"models"/args.run_name
    figure_dir=ROOT/"results/figures"/args.run_name
    split_dir=ROOT/"data/processed"/args.run_name
    protected=[p for folder in ["data/raw","data/processed"] for p in (ROOT/folder).rglob("*") if p.is_file()]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    for folder in [directory,model_dir,figure_dir,split_dir]:
        folder.mkdir(parents=True,exist_ok=True)
    provenance=prepare(directory,split_dir)
    train=pd.read_csv(split_dir/"train.csv")
    validation=pd.read_csv(split_dir/"validation.csv")
    specs={"majority":{"strategy":"most_frequent","random_state":SEED},
           "random_forest":{"n_estimators":300,"min_samples_leaf":3,"class_weight":"balanced","random_state":SEED,"n_jobs":1},
           "svm":{"C":1.,"kernel":"rbf","gamma":"scale","class_weight":"balanced","random_state":SEED}}
    manifest={"seed":SEED,"features":FEATURES,"configs":specs,"selection_metric":"validation Macro-F1; ties balanced_accuracy, accuracy, fixed insertion order",
              "standardization_fit":"training only; SVM only","resampling":False,"validation_refit":False,
              "test_used_for_selection":False,"provenance":provenance,"protected_input_sha256":before,
              "versions":{"sklearn":sklearn.__version__,"pandas":pd.__version__,"numpy":np.__version__,"joblib":joblib.__version__},
              "script_sha256":{"train.py":sha(Path(__file__)),"evaluate.py":sha(ROOT/"src/evaluate.py")}}
    (directory/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    estimators={"majority":DummyClassifier(**specs["majority"]),"random_forest":RandomForestClassifier(**specs["random_forest"]),"svm":SVC(**specs["svm"])}
    results=[]; models={}
    for name,estimator in estimators.items():
        selector=ColumnTransformer([("co2",StandardScaler() if name=="svm" else "passthrough",FEATURES)],remainder="drop")
        pipeline=Pipeline([("features",selector),("classifier",estimator)])
        pipeline.fit(train[FEATURES],train.occupancy_class)
        prediction=pipeline.predict(validation[FEATURES])
        result=metrics(validation.occupancy_class,prediction)
        save_metrics(result,name+"_validation",directory,figure_dir)
        write_csv(pd.DataFrame({"source_row":validation.source_row,"true_class":validation.occupancy_class,"predicted_class":prediction}),directory/f"{name}_validation_predictions.csv")
        path=model_dir/f"{name}.joblib"; joblib.dump(pipeline,path); models[name]=path
        results.append(dict(model=name,**{k:result[k] for k in ["accuracy","macro_f1","balanced_accuracy"]},
                            empty_precision=result["per_class"]["empty (0)"]["precision"],empty_recall=result["per_class"]["empty (0)"]["recall"],empty_f1=result["per_class"]["empty (0)"]["f1-score"]))
    comparison=pd.DataFrame(results); write_csv(comparison,directory/"validation_comparison.csv")
    winner=max(results,key=lambda r:(r["macro_f1"],r["balanced_accuracy"],r["accuracy"]))["model"]
    if before!={str(p.relative_to(ROOT)):sha(p) for p in protected}:
        raise RuntimeError("An existing raw/processed input changed")
    selection={"selected_model":winner,"selected_model_path":str(models[winner].relative_to(ROOT)),
               "selected_model_sha256":sha(models[winner]),"validation_comparison_sha256":sha(directory/"validation_comparison.csv"),
               "test_path":str((split_dir/"test.csv").relative_to(ROOT)),"test_sha256":sha(split_dir/"test.csv"),
               "decision_basis":"validation metrics only; fixed configurations; no test predictions made by train.py"}
    (directory/"selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
    print(comparison.to_string(index=False)); print("Selected on validation:",winner)


if __name__=="__main__":
    main()
