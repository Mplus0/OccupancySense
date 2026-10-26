"""Audit C2 provenance, repair evidence and candidate splits before training.

No fitting, prediction, dataset mutation or formal split creation.
Exit 0 = checks pass; exit 2 = training blocked, see saved report.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import table
from preprocess import sha, write_csv, classes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/metrics/pretraining_audit"
C2 = ROOT / "results/metrics/c2_preprocessing"


def main():
    protected = [p for folder in ["data/raw", "data/processed", "models"]
                 for p in (ROOT/folder).rglob("*") if p.is_file()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    raw = pd.read_csv(ROOT/"data/raw/ieq-dataset.csv")
    data = pd.read_csv(ROOT/"data/processed/A_C2_CO2_v1.csv")
    plan_path = C2/"candidate_A_C2_CO2_v1_candidate_session_groups.csv"
    plan = pd.read_csv(plan_path)
    repairs = pd.read_csv(C2/"repair_decisions.csv")
    manifest = json.loads((C2/"manifest.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=detail))
    check("source_hashes_match_preprocessing", all(sha(ROOT/p)==h for p,h in manifest["protected_sha256"].items()), "A/B raw and D_core vs preprocessing manifest")
    check("single_feature_dataset_hash", sha(ROOT/"data/processed/A_C2_CO2_v1.csv")==manifest["output_sha256"]["A_C2_CO2_v1"], "C2 single feature version unchanged")
    check("unique_valid_source_rows", data.source_row.is_unique and data.source_row.between(2,len(raw)+1).all(), "source_row must identify one raw record")
    source = raw.iloc[data.source_row.to_numpy()-2].reset_index(drop=True)
    check("CO2C2_matches_raw", np.array_equal(data.CO2C2.to_numpy(),source.CO2C2.to_numpy()), "No rescaling or imputation in the predictor")
    check("Occupancy_matches_raw", np.array_equal(data.Occupancy.to_numpy(),source.Occupancy.to_numpy()), "No rewritten occupancy labels")
    check("four_class_mapping_matches_raw", np.array_equal(data.occupancy_class.to_numpy(),classes(source.Occupancy).to_numpy()), "0 / 1-5 / 6-15 / >=16")
    for col in ["Datetime","Aula","Measure_ID"]:
        check(col+"_matches_raw", data[col].fillna("").tolist()==source[col].fillna("").tolist(), "Original metadata preserved")
    check("predictor_schema", manifest["default_features"]["A_C2_CO2_v1"]==["CO2C2"] and not {"TairC2","RHC2"}.intersection(data.columns), "Only CO2C2; temperature/humidity not in dataset")
    check("plan_complete_unique", plan.Measure_ID.is_unique and set(plan.Measure_ID)==set(data.Measure_ID), "Candidate plan covers every Measure_ID once")
    joined = data.merge(plan[["Measure_ID","proposed_split"]],on="Measure_ID",validate="many_to_one")
    for key in ["Measure_ID","candidate_session","candidate_leakage_group",["Datetime","CO2C2"]]:
        counts = joined.groupby(key,dropna=False).proposed_split.nunique()
        check("no_cross_split_"+str(key), counts.le(1).all(), f"crossing keys={int(counts.gt(1).sum())}")
    links = pd.read_csv(C2/"source_links.csv")
    split_map = plan.set_index("Measure_ID").proposed_split.to_dict()
    crossings = [r for r in links.to_dict("records") if r["left"] in split_map and r["right"] in split_map and split_map[r["left"]]!=split_map[r["right"]]]
    check("raw_C1_C2_source_links_not_crossed", not crossings, f"crossing links={len(crossings)}")

    # Independently reconstruct conflicts on the exact raw three-channel key,
    # not on CO2 alone; CO2 alone being non-injective is not a label error.
    key=["Datetime","CO2C2","TairC2","RHC2"]
    known=raw.Occupancy.notna() & np.isfinite(raw.Occupancy) & raw.Occupancy.between(0,60) & raw.Occupancy.mod(1).eq(0)
    raw_class=pd.Series(pd.NA,index=raw.index,dtype="Int64")
    raw_class.loc[known]=classes(raw.loc[known,"Occupancy"]).astype("Int64")
    conflict_rows=[]; conflict_groups=[]; number=0
    duplicate = raw[key].notna().all(axis=1)&raw.duplicated(key,keep=False)
    for values, block in raw.loc[duplicate].groupby(key,sort=True):
        if raw_class.loc[block.index].nunique()<=1:
            continue
        number+=1
        retained = block.index.intersection(data.source_row.to_numpy()-2)
        conflict_groups.append(dict(conflict_id=number,Datetime=values[0],CO2C2=values[1],TairC2=values[2],RHC2=values[3],
                                    raw_rows=len(block),known_classes=";".join(map(str,sorted(raw_class.loc[block.index].dropna().unique()))),
                                    retained_rows=len(retained),retained_classes=raw_class.loc[retained].nunique(),
                                    measures=";".join(sorted(block.Measure_ID.unique()))))
        for i,row in block.iterrows():
            conflict_rows.append(dict(conflict_id=number,source_row=int(i)+2,retained_in_C2=(i in retained),
                proposed_split=split_map.get(row.Measure_ID,"not_assigned"),Measure_ID=row.Measure_ID,Aula=row.Aula,
                Datetime=row.Datetime,CO2C2=row.CO2C2,TairC2=row.TairC2,RHC2=row.RHC2,
                Occupancy=row.Occupancy,occupancy_class=raw_class.loc[i]))
    evidence=pd.DataFrame(conflict_rows)
    group_evidence=pd.DataFrame(conflict_groups)
    retained_conflicts=evidence[evidence.retained_in_C2]
    reconstructed=set(retained_conflicts.source_row)
    check("conflict_flags_reproduce", reconstructed==set(data.loc[data.class_conflict_review,"source_row"]), f"retained conflict flags={len(reconstructed)}")
    check("unresolved_source_label_conflicts", len(reconstructed)==0,
          f"{len(reconstructed)} retained rows / {int(group_evidence.retained_rows.gt(0).sum())} raw sensor-time keys; label/source attribution unresolved")
    write_csv(evidence,OUT/"conflict_row_evidence.csv")
    write_csv(group_evidence,OUT/"conflict_keys.csv")
    conflict_summary=retained_conflicts.groupby(["proposed_split","Measure_ID","occupancy_class"]).size().reset_index(name="rows")
    write_csv(conflict_summary,OUT/"conflict_by_measure_class.csv")

    # Independently validate every recorded applied repair against raw data.
    applied=repairs[repairs.action.eq("divide_1000_supported_candidate")].copy()
    repair_findings=[]
    for r in applied.itertuples():
        row=raw.iloc[int(r.source_row)-2]; anchor=raw.iloc[int(r.past_anchor_source_row)-2]
        lo,hi=(-20,60) if r.field=="TairC2" else (0,100)
        ref_field="TairD" if r.field=="TairC2" else "RHD"
        tolerance=2. if r.field=="TairC2" else 5.
        age=(pd.Timestamp(row.Datetime)-pd.Timestamp(anchor.Datetime)).total_seconds()
        x=float(row[r.field]); ref=float(row[ref_field]); past=float(anchor[r.field])
        divisors=[f for f in [1,10,100,1000,10000] if lo<=x/f<=hi and abs(x/f-ref)<=tolerance and abs(x/f-past)<=tolerance]
        ok=(r.field in ["TairC2","RHC2"] and r.source_md5_verified and row.Measure_ID==anchor.Measure_ID
            and np.isfinite(ref) and lo<=ref<=hi and lo<=past<=hi and 0<age<=600
            and not lo<=x<=hi and abs(x)>=1000 and divisors==[1000]
            and np.isclose(r.original_value,x) and np.isclose(r.output_value,x/1000)
            and np.isclose(r.past_anchor_value,past) and np.isclose(r.reference_value,ref)
            and np.isclose(r.past_age_seconds,age))
        split=split_map.get(row.Measure_ID,"not_assigned")
        anchor_split=split_map.get(anchor.Measure_ID,"not_assigned")
        repair_findings.append(dict(source_row=r.source_row,field=r.field,evidence_matches_raw=bool(ok),
            proposed_split=split,anchor_split=anchor_split,cross_split_reference=split!=anchor_split,
            in_single_feature_version=r.source_row in set(data.source_row)))
    repair_checks=pd.DataFrame(repair_findings)
    check("repair_evidence_matches_raw", repair_checks.evidence_matches_raw.all(), f"{len(applied)} field operations independently checked; not publisher-confirmed fixes")
    check("repair_references_stay_within_session", not repair_checks.cross_split_reference.any(), "Past same-Measure_ID anchors; D reference from same raw row")
    write_csv(repair_checks,OUT/"repair_evidence_checks.csv")
    repair_summary=applied.groupby("field").agg(field_operations=("source_row","size"),raw_rows=("source_row","nunique"),
                                               rows_in_single=("single_retained","sum"),rows_in_multi=("multi_retained","sum")).reset_index()
    write_csv(repair_summary,OUT/"repair_summary.csv")
    # Multi-version labels must also be unchanged, although that version is excluded.
    multi=pd.read_csv(ROOT/"data/processed/A_C2_core_supported_candidate_v1.csv")
    check("multi_version_labels_unchanged",np.array_equal(multi.Occupancy.to_numpy(),raw.iloc[multi.source_row.to_numpy()-2].Occupancy.to_numpy()),"Temperature/humidity operations never rewrite occupancy")
    split_summary=[]
    for split in ["train","validation","test"]:
        block=joined[joined.proposed_split.eq(split)]
        row=dict(split=split,rows=len(block),measure_groups=block.Measure_ID.nunique(),candidate_sessions=block.candidate_session.nunique(),
                 schools=block.school_candidate.nunique(),conflict_rows=int(block.source_row.isin(reconstructed).sum()))
        for c in range(4):
            subset=block[block.occupancy_class.eq(c)]
            row[f"class_{c}"]=len(subset); row[f"class_{c}_sessions"]=subset.candidate_session.nunique()
        split_summary.append(row)
    split_summary=pd.DataFrame(split_summary)
    write_csv(split_summary,OUT/"candidate_split_audit.csv")
    unchanged=before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    check("protected_data_and_models_unchanged",unchanged,"No datasets, existing models or raw files changed")
    check_frame=pd.DataFrame(checks); write_csv(check_frame,OUT/"checks.csv")
    failed=check_frame.loc[~check_frame.passed,"check"].tolist()
    status="BLOCKED" if failed else "PASS"
    metadata={"status":status,"blockers":failed,"model_training_performed":False,"formal_split_created":False,
              "feature_allowlist":["CO2C2"],"protected_sha256":before,"candidate_plan_sha256":sha(plan_path),
              "repair_log_sha256":sha(C2/"repair_decisions.csv"),"script_sha256":sha(Path(__file__)),
              "conflict_retained_rows":len(reconstructed),"conflict_retained_keys":int(group_evidence.retained_rows.gt(0).sum()),
              "repair_field_operations":len(applied),"repair_distinct_source_rows":int(applied.source_row.nunique())}
    (OUT/"audit_manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    example=retained_conflicts[retained_conflicts.conflict_id.eq(retained_conflicts.conflict_id.iloc[0])] if len(retained_conflicts) else retained_conflicts
    report=f"""# 第一轮基线训练前审查：{status}

## 决定

训练门禁状态 **{status}**。用户要求发现严重问题先停止训练。本次发现未解决的同时间、同C2三通道观测与类别标注冲突，因此停止在审查阶段；未建立正式三集、未编写或执行训练/评估入口、未拟合多数类/随机森林/SVM，也没有模型对比数值。未用测试集做模型选择；此次查看其分组和标签仅用于完整性审核。

## 368条冲突是什么

从原始CSV重新按 Datetime+CO2C2+TairC2+RHC2 完全相同的键重建冲突，保留数据中有 **{len(reconstructed)} 条**涉及 **{int(group_evidence.retained_rows.gt(0).sum())} 个观测键**。预处理标记与本次重算一致，不是温湿度修复引入。这里不是把“CO₂相同但人数不同”误当错误：同一浓度可对应多人数组合；阻断依据是多通道、时间、长段复制及测量来源归属同时存在歧义，尚不能确定哪一条人数属于该传感器观测。

原始证据示例：

{table(example[['source_row','Measure_ID','Aula','Datetime','CO2C2','TairC2','RHC2','Occupancy','occupancy_class']])}

全部键、原始行和逐组类别统计分别见 conflict_keys.csv、conflict_row_evidence.csv、conflict_by_measure_class.csv。可疑重复同时包含中/高或其他不同类别；不能按多数票、保留第一条、按模型预测重标或仅删除不利于分数的记录。原始文件已经如此，现有清洗没有新增错误标签，但继承了未解决的来源/标签问题。

## 3,136次修复的影响

{table(repair_summary)}

共 {len(applied)} 次**字段值操作**，涉及 {applied.source_row.nunique()} 个原始记录，不等于新增或修改3,136个人数标签。已逐项重算原值、原始历史锚点、D参考、时间差、容差及唯一可行因子；规则证据是否通过见 checks.csv。这只能证明实现符合固定候选规则，不能证明发布者确认了其物理正确性。

单CO₂版本的每条CO2C2、Occupancy、四分类及时间/教室/ID均逐行比对原始文件；预测字段未修复，标签未改写，温湿度不在单特征数据中。修复只写入另一个候选版本，参考来自同组过去观测或同一原始行，没有跨候选集合取值；本轮明确禁用该多传感器候选。

## 候选会话划分审核（尚未正式采用）

{table(split_summary)}

同一Measure_ID、candidate_session、继承依赖组和相同时间+CO₂键，以及已知原始C1/C2同源关联，均检查是否跨集合；结果见 checks.csv。**即使未发现这些已知关系跨集，也不能消除集合内部标签冲突。** 当前会话数量是保守候选连通组，不是来源方确认的独立实验数。会话方案允许同校/教室跨日期进入不同集合，评估目标只能是已知场景分布下的新批次，不是严格新学校泛化；时间也不是严格未来留出。无人类每个验证/测试集合仅来自2个候选会话，分钟样本不能当成独立采集事件。

## 下一步

先核查冲突ID的课堂人数记录、设备位置/复制轨迹和真实时间。来源确认后形成明确、可追溯的纠错规则。如果暂时无法溯源，可另行审核“整段隔离有歧义的同源会话”的新版本，并重新检查类别和会话覆盖；此次未自动删除或重标，也没有自行换数据绕过门禁。标签问题解决前，不进行正式训练及模型比较。

复现：`python src/check_training_readiness.py`。阻断时返回退出码2是预期行为。原始/清洗数据与已有模型哈希保持不变；全部检查和输入指纹见checks.csv与audit_manifest.json。
"""
    (OUT/"training_readiness_report.md").write_text(report,encoding="utf-8")
    print("Training gate:",status,"; blockers:",failed)
    print(split_summary.to_string(index=False)); print(repair_summary.to_string(index=False))
    print("Conflict keys:",int(group_evidence.retained_rows.gt(0).sum()))
    return 2 if failed else 0


if __name__=="__main__":
    raise SystemExit(main())
