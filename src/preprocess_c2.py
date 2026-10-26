"""Prepare reviewable C2 versions; no model fitting or final dataset split.

Source metadata verified at https://zenodo.org/records/14621677 on 2026-10-26.
Repair rules are conservative engineering inferences, not a publisher erratum.
"""
from datetime import date
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

from audit_data import table
from preprocess import clean, classes, CLASS_NAMES, sha, write_csv
from analyze_split import components, school

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/metrics/c2_preprocessing"
PROCESSED = ROOT / "data/processed"
SOURCE_MD5 = "4e79072861dc6c0ec9d5035769fbfb8c"
AS_OF = date(2026, 10, 26)
SPECS = {"CO2C2": (250, 5000), "TairC2": (-20, 60), "RHC2": (0, 100)}
DIVISORS = [1, 10, 100, 1000, 10000]


def supported_repairs(a, source_verified):
    """Past-only, original-scale anchors; never propagate repaired values.

    Both a raw C2 observation <=10 minutes earlier in the same Measure_ID
    and simultaneous raw D must support exactly one of the tested divisors.
    Tolerance 2 degC / 5 RH percentage points is fixed, not fitted on this data.
    """
    times = pd.to_datetime(a.Datetime, errors="coerce").astype("datetime64[us]")
    values = a[["TairC2", "RHC2"]].copy()
    records = []
    for col, reference, tolerance in [("TairC2", "TairD", 2.), ("RHC2", "RHD", 5.)]:
        lo, hi = SPECS[col]
        raw = pd.to_numeric(a[col], errors="coerce")
        normal = np.isfinite(raw) & raw.between(lo, hi)
        candidate = np.isfinite(raw) & ~normal & raw.abs().ge(1000) & (raw/1000).between(lo,hi)
        for measure, block in a.groupby("Measure_ID", sort=True):
            last = None
            for i in times.loc[block.index].sort_values().index:
                if normal.at[i] and pd.notna(times.at[i]):
                    last = i
                    continue
                if not candidate.at[i]:
                    continue
                age = (times.at[i]-times.at[last]).total_seconds() if last is not None and pd.notna(times.at[i]) else np.nan
                anchor_ok = np.isfinite(age) and 0 < age <= 600
                ref = pd.to_numeric(pd.Series([a.at[i, reference]]), errors="coerce").iloc[0]
                ref_ok = np.isfinite(ref) and lo <= ref <= hi
                admissible = []
                if anchor_ok and ref_ok:
                    admissible = [factor for factor in DIVISORS
                                  if lo <= raw.at[i]/factor <= hi
                                  and abs(raw.at[i]/factor-raw.at[last]) <= tolerance
                                  and abs(raw.at[i]/factor-ref) <= tolerance]
                applied = bool(source_verified and admissible == [1000])
                if applied:
                    values.at[i,col] = raw.at[i]/1000
                records.append(dict(source_row=int(i)+2, Measure_ID=measure, field=col,
                    original_value=raw.at[i], candidate_value=raw.at[i]/1000,
                    output_value=values.at[i,col], source_md5_verified=source_verified,
                    past_anchor_source_row=int(last)+2 if last is not None else None,
                    past_anchor_value=raw.at[last] if last is not None else None,
                    past_age_seconds=age, past_anchor_valid=bool(anchor_ok),
                    reference_field=reference, reference_value=ref, reference_valid=bool(ref_ok),
                    tolerance=tolerance, admissible_divisors=";".join(map(str,admissible)),
                    action="divide_1000_supported_candidate" if applied else "UNRESOLVED_NO_CHANGE",
                    rule_id="C2_SCALE_V1_SOURCE_PAST_D_UNIQUE"))
    return values, pd.DataFrame(records)


def candidate_split(frame, level):
    """Whole units, >=20 rows and >=2 conservative sessions per class/subset."""
    counts = pd.crosstab(frame[level], frame.occupancy_class).reindex(columns=range(4),fill_value=0)
    support = frame.groupby([level,"occupancy_class"]).candidate_session.nunique().unstack(fill_value=0).reindex(index=counts.index,columns=range(4),fill_value=0)
    data = np.column_stack([counts.sum(axis=1), counts.to_numpy()])
    total = data.sum(axis=0); g=len(data); n=3*g+15
    if g<3 or (total[1:]<60).any():
        return None, {"status":"infeasible_by_counts", "minimum_class_rows":20,"minimum_class_sessions":2}
    c=np.r_[np.zeros(3*g),np.tile(np.array([2,1,1,1,1])/total,3)]
    matrix=[]; lower=[]; upper=[]
    def add(row,lo,hi):
        matrix.append(row); lower.append(lo); upper.append(hi)
    for i in range(g):
        row=np.zeros(n); row[3*i:3*i+3]=1; add(row,1,1)
    for s,ratio in enumerate([.7,.15,.15]):
        for k in range(5):
            row=np.zeros(n); row[s:3*g:3]=data[:,k]
            if k:
                add(row.copy(),20,np.inf)
                supp=np.zeros(n); supp[s:3*g:3]=support[k-1]; add(supp,2,np.inf)
            row[3*g+s*5+k]=-1; add(row.copy(),-np.inf,ratio*total[k])
            row[3*g+s*5+k]=1; add(row.copy(),ratio*total[k],np.inf)
    result=milp(c,integrality=np.r_[np.ones(3*g),np.zeros(15)],
        bounds=Bounds(np.zeros(n),np.r_[np.ones(3*g),np.full(15,np.inf)]),
        constraints=LinearConstraint(matrix,lower,upper),options={"time_limit":20,"mip_rel_gap":.001})
    info={"status":int(result.status),"message":result.message,"minimum_class_rows":20,"minimum_class_sessions":2,
          "objective":float(result.fun) if result.fun is not None else None,
          "mip_gap":float(result.mip_gap) if getattr(result,"mip_gap",None) is not None else None}
    if result.x is None:
        return None,info
    labels=np.array(["train","validation","test"])[result.x[:3*g].reshape(g,3).argmax(axis=1)]
    return dict(zip(counts.index,labels)),info


def main():
    raw_path=ROOT/"data/raw/ieq-dataset.csv"
    protected=[*sorted((ROOT/"data/raw").glob("*")),PROCESSED/"A_D_core_v1.csv"]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    for directory in [OUT,PROCESSED]:
        if (ROOT/"data/raw").resolve() in [directory.resolve(),*directory.resolve().parents]:
            raise ValueError("Refusing raw output directory")
        directory.mkdir(parents=True,exist_ok=True)
    a=pd.read_csv(raw_path)
    verified=hashlib.md5(raw_path.read_bytes()).hexdigest()==SOURCE_MD5
    repaired,repair_log=supported_repairs(a,verified)
    # Reuse raw-source dependency evidence; do not invoke preprocess.main().
    _,_,_,review,_,dep_map=clean(a,AS_OF)
    times=pd.to_datetime(a.Datetime,errors="coerce")
    y=pd.to_numeric(a.Occupancy,errors="coerce")
    raw_meta=a[["Datetime","Aula","Measure_ID"]].copy()
    raw_meta.insert(0,"source_row",np.arange(len(a))+2)
    raw_meta["school_candidate"]=a.Measure_ID.map(school)
    links=[]; match_rows=[]
    for members in pd.Series(dep_map).groupby(pd.Series(dep_map)):
        links.extend(combinations(members[1].index,2))
    conflict=pd.Series(False,index=a.index)
    for sensor in [1,2]:
        cols=["Datetime",f"CO2C{sensor}",f"TairC{sensor}",f"RHC{sensor}"]
        mask=a[cols].notna().all(axis=1)&a.duplicated(cols,keep=False)
        for _,block in a.loc[mask].groupby(cols):
            known=pd.to_numeric(block.Occupancy,errors="coerce").dropna()
            known=known[np.isfinite(known)&known.between(0,60)&known.mod(1).eq(0)]
            bad=classes(known).nunique()>1
            if sensor==2 and bad:
                conflict.loc[block.index]=True
            for left,right in combinations(sorted(block.Measure_ID.unique()),2):
                links.append((left,right)); match_rows.append(dict(sensor=sensor,left=left,right=right,matched_keys=1,class_conflict_keys=int(bad)))
    write_csv(pd.DataFrame(match_rows).groupby(["sensor","left","right"],as_index=False).sum(),OUT/"source_links.csv")
    # Also bind equal predictor+time keys in each proposed representation.
    # Single-channel coincidences are conservative constraints, not proof of identity.
    for frame,cols in [(a,["Datetime","CO2C2"]),
                       (a.assign(TairC2=repaired.TairC2,RHC2=repaired.RHC2),["Datetime","CO2C2","TairC2","RHC2"])]:
        for _,b in frame[frame[cols].notna().all(axis=1)&frame.duplicated(cols,keep=False)].groupby(cols):
            links.extend(combinations(sorted(b.Measure_ID.unique()),2))
    for _,block in raw_meta.groupby(["school_candidate",times.dt.date]):
        links.extend(combinations(sorted(block.Measure_ID.unique()),2))
    session_map=components(sorted(a.Measure_ID.unique()),links)
    raw_meta["candidate_session"]=a.Measure_ID.map(session_map)
    school_links=list(links)
    for _,block in raw_meta.groupby("school_candidate"):
        school_links.extend(combinations(sorted(block.Measure_ID.unique()),2))
    raw_meta["school_unit"]=a.Measure_ID.map(components(sorted(a.Measure_ID.unique()),school_links))
    raw_meta["candidate_leakage_group"]=a.Measure_ID.map(dep_map)
    raw_meta["class_conflict_review"]=conflict
    raw_meta["review_required"]=review.any(axis=1)|conflict
    raw_meta["date_measure_mismatch"]=review.date_measure_mismatch

    masks={
        "missing_label":a.Occupancy.isna(),
        "invalid_label":a.Occupancy.notna()&~(np.isfinite(y)&y.between(0,60)&y.mod(1).eq(0)),
        "invalid_time":times.isna(), "future_time":times.dt.date.gt(AS_OF),
        "missing_CO2C2":a.CO2C2.isna(),
        "invalid_CO2C2":a.CO2C2.notna()&~np.isfinite(a.CO2C2),
        "CO2C2_out_of_range":np.isfinite(a.CO2C2)&~a.CO2C2.between(*SPECS["CO2C2"])}
    flags=pd.DataFrame(masks).fillna(False)
    keep=pd.Series(True,index=a.index); reasons=pd.Series("",index=a.index); steps=[]
    for name,mask in flags.items():
        dropped=keep&mask; reasons.loc[dropped]=name
        steps.append(dict(rule=name,matched_raw_rows=int(mask.sum()),removed_this_step=int(dropped.sum()),remaining=int((keep&~mask).sum())))
        keep &= ~mask
    base=raw_meta.copy(); base["Occupancy"]=y
    base["CO2C2"]=a.CO2C2
    base.loc[keep,"occupancy_class"]=classes(y.loc[keep])
    base["occupancy_class_name"]=base.occupancy_class.map(dict(enumerate(CLASS_NAMES)))
    base.loc[keep,"Occupancy"]=y.loc[keep].astype(int)
    single=base.loc[keep].copy()
    single["occupancy_class"]=single.occupancy_class.astype(int)
    single["Occupancy"]=single.Occupancy.astype(int)
    multi_ok=keep.copy(); original_ok=keep.copy()
    for c in ["TairC2","RHC2"]:
        original_ok &= np.isfinite(a[c])&a[c].between(*SPECS[c])
        multi_ok &= np.isfinite(repaired[c])&repaired[c].between(*SPECS[c])
    multi=single.loc[multi_ok.loc[single.index]].copy()
    for c in ["TairC2","RHC2"]:
        multi[c]=repaired.loc[multi.index,c]
        multi[c+"_raw"]=a.loc[multi.index,c]
        multi[c+"_repaired"]=multi[c].ne(multi[c+"_raw"])
    raw_only=single.loc[original_ok.loc[single.index]].copy()
    raw_only[["TairC2","RHC2"]]=a.loc[raw_only.index,["TairC2","RHC2"]]
    versions={"A_C2_CO2_v1":single,"A_C2_core_raw_v1":raw_only,"A_C2_core_supported_candidate_v1":multi}
    logs=pd.concat([raw_meta,flags],axis=1)
    logs["single_retained"]=keep; logs["single_exclusion_reason"]=reasons
    logs["multi_retained"]=multi_ok
    for c in ["TairC2","RHC2"]:
        logs[c+"_unresolved"]=~(np.isfinite(repaired[c])&repaired[c].between(*SPECS[c]))
    write_csv(logs,OUT/"row_decisions.csv")
    write_csv(pd.DataFrame(steps),OUT/"processing_steps.csv")
    repair_log["single_retained"]=repair_log.source_row.isin(single.source_row)
    repair_log["multi_retained"]=repair_log.source_row.isin(multi.source_row)
    write_csv(repair_log,OUT/"repair_decisions.csv")
    write_csv(pd.concat([raw_meta,a.drop(columns=["Datetime","Aula","Measure_ID"])],axis=1).loc[raw_meta.review_required],OUT/"manual_review.csv")

    summaries=[]; split_summaries=[]; solvers={}; outputs={}
    for name,frame in versions.items():
        path=PROCESSED/(name+".csv"); write_csv(frame,path); outputs[name]=sha(path)
        features=["CO2C2"] if name=="A_C2_CO2_v1" else ["CO2C2","TairC2","RHC2"]
        count=frame.occupancy_class.value_counts().reindex(range(4),fill_value=0)
        summaries.append(dict(version=name,rows=len(frame),measure_groups=frame.Measure_ID.nunique(),
            candidate_sessions=frame.candidate_session.nunique(),schools=frame.school_candidate.nunique(),
            conflict_review_rows=int(frame.class_conflict_review.sum()),**{f"class_{i}":int(count[i]) for i in range(4)}))
        for group in ["Measure_ID","candidate_session","school_candidate"]:
            ct=pd.crosstab(frame[group],frame.occupancy_class).reindex(columns=range(4),fill_value=0)
            ct.columns=[f"class_{i}" for i in range(4)]
            write_csv(ct.reset_index(),OUT/f"{name}_counts_{group}.csv")
        if name=="A_C2_core_raw_v1":
            continue
        for level in ["school_unit","candidate_session"]:
            mapping,info=candidate_split(frame,level); solvers[name+"/"+level]=info
            if mapping is None:
                continue
            proposal=frame.assign(proposed_split=frame[level].map(mapping))
            assert proposal.groupby("candidate_session").proposed_split.nunique().max()==1
            assert proposal.groupby(["Datetime",*features]).proposed_split.nunique().max()==1
            groups=proposal.groupby("Measure_ID").agg(proposed_split=("proposed_split","first"),rows=("source_row","size"),
                school=("school_candidate","first"),session=("candidate_session","first"),start=("Datetime","min"),end=("Datetime","max"))
            write_csv(groups.reset_index(),OUT/f"candidate_{name}_{level}_groups.csv")
            for split,block in proposal.groupby("proposed_split"):
                ct=block.occupancy_class.value_counts().reindex(range(4),fill_value=0)
                support=block.groupby("occupancy_class").candidate_session.nunique().reindex(range(4),fill_value=0)
                assert ct.min()>=20 and support.min()>=2
                split_summaries.append(dict(version=name,policy=level,split=split,rows=len(block),fraction=len(block)/len(frame),
                    measure_groups=block.Measure_ID.nunique(),sessions=block.candidate_session.nunique(),
                    **{f"class_{i}":int(ct[i]) for i in range(4)},**{f"class_{i}_sessions":int(support[i]) for i in range(4)}))
    summary=pd.DataFrame(summaries); splits=pd.DataFrame(split_summaries)
    write_csv(summary,OUT/"version_summary.csv"); write_csv(splits,OUT/"candidate_split_summary.csv")
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    manifest={"status":"CANDIDATES_FOR_REVIEW_NOT_FINAL_SPLIT", "as_of":str(AS_OF),"protected_sha256":before,
        "source_md5":hashlib.md5(raw_path.read_bytes()).hexdigest(),"published_md5_match":verified,
        "source_url":"https://zenodo.org/records/14621677", "paper_url":"https://doi.org/10.1155/ina/5517242",
        "source_checked_on":"2026-10-26", "source_units":{"TairC2":"degC","RHC2":"%RH","CO2C2":"ppm"},
        "screening_bounds":SPECS,"co2_lower_bound_note":"250 is project screening assumption, not instrument minimum",
        "repair_rule":"fixed past-only original C2 within 600 seconds and simultaneous raw D; tolerances 2 degC/5 RH points; divisor1000 must uniquely satisfy both references among [1,10,100,1000,10000]; source checksum must match",
        "repair_status":"corroborated engineering candidate, not publisher-confirmed erratum; unresolved unchanged",
        "output_sha256":outputs,"default_features":{n:["CO2C2"] if n=="A_C2_CO2_v1" else ["CO2C2","TairC2","RHC2"] for n in versions},
        "metadata_excluded_from_features":"all remaining columns, including raw values, repair flags, labels, IDs, time and review flags",
        "solvers":solvers,"script_sha256":sha(Path(__file__)),"no_training":True,"no_final_split":True,
        "dependencies_sha256":{n:sha(ROOT/"src"/n) for n in ["audit_data.py","preprocess.py","analyze_split.py"]},
        "versions":{"pandas":pd.__version__,"numpy":np.__version__}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    applied=repair_log.action.eq("divide_1000_supported_candidate")
    report=f"""# C2 数据准备与质量报告

## 数据版本（均待审核）

{table(summary)}

class_0/1/2/3分别为0、1~5、6~15、≥16人。默认特征白名单见manifest；时间、学校、教室、测量ID、会话、原值及修复/审核标记不作为预测特征。候选会话为同校同日及同源依赖的保守连通组，不是已证明独立的真实会话。

## 来源核实与量纲

本地CSV MD5与[Zenodo原始发布文件](https://zenodo.org/records/14621677)一致：`{SOURCE_MD5}`，异常已存在于发布数据。其字段说明确认C2为S12，温度°C、湿度%、CO₂ ppm。[原论文表2](https://doi.org/10.1155/ina/5517242)列出S1温度−20~60°C、湿度0~100%、CO₂ 0~5000ppm；本版CO₂筛查采用250~5000，250仍为项目下限假设。未找到发布者明确的逐行千倍修复勘误，**单位确认不等于所有异常可统一缩放**。论文同时提醒传感器温度差异及位置效应，不能仅以D设备接近为修复依据。

CO2C2不做缩放；标签/时间/CO₂异常按固定规则排除，完整互斥处理数见processing_steps.csv。DewPointC2等未选字段不修复、不加入预测输入。已知CO₂巨大值继续排除，不因温湿度修复推断CO₂也可修复。

## 温湿度的保守修复规则

只考虑原值越界且÷1000进入仪器范围的候选；发布文件校验匹配，并且同时满足：① 同一Measure_ID中此前10分钟内存在**原始正常尺度**的同通道观测；② 同时刻D设备对应通道原值有效；③ 缩放结果与两项参考差均≤2°C或≤5个百分点；④ 在1/10/100/1000/10000各因子中仅1000满足上述条件。阈值为预先固定的保守工程规则，不是论文给出的修复标准；未用类别/全量统计拟合阈值。仅向过去找参考，不用未来观测、不跨测量组、不用已修复值递推。

共评估 {len(repair_log):,} 个温湿度字段异常候选；有 {int(applied.sum()):,} 个字段值满足规则，其余 {int((~applied).sum()):,} 个保留原值待核实。仅完整通过的行进入多传感器候选；单CO₂版本不受温湿度修复影响。日志repair_decisions.csv逐项保留原值、假设值、输出值、参考行号/值、时间差、容差、可接受因子及是否进入候选。候选修复不是对真实温湿度的认证，后续仍应抽样核对设备原始导出；当前多传感器版本依赖D参考及历史锚点，覆盖具有选择偏差，不能据此承诺仅靠C2在线复现相同修复能力。

## 同源与标签质量

继承原始全数据的重复依赖（含被排除行形成的桥），重新核查C1/C2三通道与时间一致的跨ID记录，再把同校同日整体绑定。相同CO₂+时间也保守绑定，属于防泄漏约束而非确认同源。日期与ID冲突及类别矛盾不擅自修正；manual_review.csv列出原始证据。即使只用CO₂，也不依据相同单值推断标签错误；class_conflict_review仅表示同时间同C2三通道原值出现不同四分类标签。可疑重复保留，未任意删一条；它们仍影响训练权重和评估，需审核。

## 划分候选

{table(splits) if len(splits) else '没有满足最低覆盖条件的方案。'}

候选优先保证整依赖/会话不跨集合；学校策略进一步隔离同校所有日期，支持新学校泛化；会话策略允许同校不同日期跨集合，只支持新批次评估。优化目标是接近70/15/15及类别比例，每集每类≥20条、且每类至少来自2个候选会话为硬约束。此最低门槛不是统计可靠性的充分条件；需另看独立学校数、重复样本、少数来源集中及置信区间。求解不可行时不放宽为随机行划分，状态见manifest。多传感器候选覆盖若不足则不输出该版本的候选分组。这里只导出组级候选归属，未导出train/validation/test数据文件，也未采纳最终方案。

建议先审核单CO₂版本的冲突标注及来源分组，将其作为更广覆盖的候选主版本；多传感器版本用于修复规则审查，不能为了扩大数量降低证据要求。后续若要比较特征增益，应在同一可信样本与同一组级划分上对比，避免把选择样本的差异当作新增特征收益。未经确认不做最终划分或训练。

## 复现及文件

运行 `python src/preprocess_c2.py`；无需联网，已核实的来源URL、校验与规则固化在脚本/manifest中；输入变化将禁用修复。输出data/processed/A_C2_*.csv；统计、逐行决定、修复日志、人工清单、组级候选和manifest位于results/metrics/c2_preprocessing/。原始A/B及D_core哈希前后一致。无插补、标准化、重采样或模型训练。
"""
    (OUT/"quality_report.md").write_text(report,encoding="utf-8")
    print(summary.to_string(index=False)); print(splits.to_string(index=False)); print("Applied field repairs:",int(applied.sum()))


if __name__=="__main__":
    main()
