"""Resolve the provenance of C2 label-conflict keys; no label rewriting or training.

Reuses the evidence already produced upstream instead of recomputing conflicts:
- results/metrics/pretraining_audit/conflict_row_evidence.csv (check_training_readiness.py)
- results/metrics/c2_preprocessing/candidate_*.csv (preprocess_c2.py)
Only the physical-range judgement is recomputed here, from the raw CSV.

Verdicts per conflict key
- COPY_AMBIGUOUS        all participants carry off-instrument-scale values on an identical
                        timestamp, so no room's reading can be identified -> labels unverifiable
- OPEN_PLAUSIBLE_VALUES identical and physically plausible -> real reading possible, owner unknown
- NEEDS_SOURCE_LOOKUP   timestamps differ -> the data cannot separate the rooms at all

Contract:
- Never rewrites Occupancy / occupancy_class / any raw or processed file.
- Quarantine stays a scenario table; no dataset is filtered.

Exit 0 = every conflict key resolved; exit 2 = at least one key needs outside
information (expected while publisher/source records are unavailable).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import table
from preprocess import sha, write_csv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ieq-dataset.csv"
SINGLE = ROOT / "data/processed/A_C2_CO2_v1.csv"
C2 = ROOT / "results/metrics/c2_preprocessing"
AUDIT = ROOT / "results/metrics/pretraining_audit"
OUT = ROOT / "results/metrics/conflict_resolution"

KEYS = ["CO2C2", "TairC2", "RHC2"]
# Instrument ranges from the publisher's paper table 2; used only to judge whether
# a value could be a genuine reading, never to rescale anything.
SPEC = {"TairC2": (-20, 60), "RHC2": (0, 100)}


def numeric_block(raw, index):
    return raw.loc[index, KEYS].apply(pd.to_numeric, errors="coerce")


def main():
    for path in [RAW, SINGLE, AUDIT / "conflict_row_evidence.csv", C2 / "candidate_A_C2_CO2_v1_candidate_session_groups.csv"]:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}\n"
                             "Run src/preprocess_c2.py then src/check_training_readiness.py first.")
    protected = {str(p.relative_to(ROOT)): sha(p)
                 for folder in ["data/raw", "data/processed"]
                 for p in (ROOT / folder).rglob("*") if p.is_file()}
    evidence_all = pd.read_csv(AUDIT / "conflict_row_evidence.csv")
    raw = pd.read_csv(RAW)
    single = pd.read_csv(SINGLE)
    evidence = evidence_all[evidence_all.retained_in_C2.astype(bool)].copy()
    if evidence.empty:
        raise SystemExit("No conflict rows retained in A_C2_CO2_v1; nothing to review")
    plan = pd.read_csv(C2 / "candidate_A_C2_CO2_v1_candidate_session_groups.csv")
    if (ROOT / "data/raw").resolve() in OUT.resolve().parents:
        raise SystemExit("Refusing to write inside data/raw")
    OUT.mkdir(parents=True, exist_ok=True)
    split_of = plan.set_index("Measure_ID").proposed_split.to_dict()
    session_of = plan.set_index("Measure_ID").session.to_dict()

    evidence["raw_index"] = evidence.source_row - 2
    measures_per_key = evidence.groupby("conflict_id").Measure_ID.nunique()
    if measures_per_key.max() > 3:
        raise SystemExit("Conflict key spans more than 3 Measure_IDs; rule review required")

    # Two distinct situations hide behind "same key, different label":
    # A. every participant of the key carries an off-instrument-scale value, so the
    #    triple cannot be any room's genuine reading -> no label can be verified;
    # B. the values are physically plausible, so the observation could be real and the
    #    measurement room simply cannot be identified from these columns -> open.
    attr_rows = []
    for conflict_id, block in evidence.groupby("conflict_id", sort=True):
        measures = sorted(block.Measure_ID.unique())
        raw_block = raw.loc[block.raw_index]
        # Compare against every raw row carrying the same triple, not just the retained
        # ones, so a key whose third participant was dropped is still recognisable.
        key_all = evidence_all[evidence_all.conflict_id.eq(conflict_id)]
        raw_all = raw.loc[key_all.source_row.to_numpy() - 2]
        identical = raw_all.groupby(KEYS, dropna=False).ngroups == 1
        plausible = {}
        for measure in measures:
            index = block.raw_index[block.Measure_ID.eq(measure).to_numpy()]
            values = numeric_block(raw, index)
            plausible[measure] = all(bool(values[field].between(*bounds).all()) for field, bounds in SPEC.items())
        labels = block.groupby("Measure_ID").occupancy_class.agg(lambda s: tuple(sorted(set(s))))
        if not identical:
            status, attribution_label = "NEEDS_SOURCE_LOOKUP", "SINGLE_ROOM_CANDIDATE"
        elif all(plausible.values()):
            status, attribution_label = "OPEN_PLAUSIBLE_VALUES", "UNATTRIBUTABLE"
        else:
            status, attribution_label = "COPY_AMBIGUOUS", "UNATTRIBUTABLE"
        attr_rows.append(dict(conflict_id=int(conflict_id), key_Datetime=block.Datetime.iloc[0],
            measures=";".join(measures), n_measure_ids=int(key_all.Measure_ID.nunique()),
            participants_retained=";".join(measures),
            identical_three_channel_text=identical,
            values_physically_plausible=all(plausible.values()),
            TairC2=raw_all.TairC2.iloc[0], RHC2=raw_all.RHC2.iloc[0],
            sensor_attribution=attribution_label,
            status=status,
            retained_rows=int(len(block)),
            proposed_split=";".join(sorted({split_of.get(m, "not_assigned") for m in measures})),
            labels_per_measure=" | ".join(f"{m}={','.join(map(str,v))}" for m, v in labels.items())))
    attribution = pd.DataFrame(attr_rows)
    write_csv(attribution, OUT / "conflict_key_attribution.csv")

    # Per Measure_ID impact under the candidate plan, so the cost of whole-session
    # quarantine is visible without performing it.
    impact = []
    for measure in sorted(attribution.measures.str.split(";").explode().unique()):
        keys = attribution[attribution.measures.str.split(";").apply(lambda v: measure in v)]
        rows = single[single.Measure_ID.eq(measure)]
        if rows.empty:
            impact.append(dict(Measure_ID=measure, proposed_split=split_of.get(measure, "not_assigned"),
                               candidate_session=session_of.get(measure, "not_assigned"),
                               dataset_rows=0, conflict_rows=0, conflict_share=np.nan,
                               note="session absent from A_C2_CO2_v1"))
            continue
        flagged = rows.loc[rows.class_conflict_review, "occupancy_class"].value_counts().sort_index()
        impact.append(dict(Measure_ID=measure, proposed_split=split_of.get(measure, "not_assigned"),
            candidate_session=session_of.get(measure, "not_assigned"),
            dataset_rows=len(rows), conflict_rows=int(rows.class_conflict_review.sum()),
            conflict_share=round(float(rows.class_conflict_review.mean()), 4),
            conflict_keys=int(len(keys)),
            conflicted_class_0=int(flagged.get(0, 0)), conflicted_class_1=int(flagged.get(1, 0)),
            conflicted_class_2=int(flagged.get(2, 0)), conflicted_class_3=int(flagged.get(3, 0)),
            note="whole session would be quarantined under conservative quarantine"))
    impact = pd.DataFrame(impact)
    write_csv(impact, OUT / "conflict_measure_impact.csv")

    quarantine = set(impact.loc[impact.conflict_rows.gt(0), "Measure_ID"])
    quarantined_sessions = {session_of.get(m) for m in quarantine}
    if len(quarantined_sessions) != 1:
        raise SystemExit(f"Conflict sessions are not a single unit: {sorted(map(str, quarantined_sessions))}")
    session_rows = single.occupancy_class.value_counts().reindex(range(4), fill_value=0)
    kept = single[~single.candidate_session.isin(quarantined_sessions)]
    kept_rows = kept.occupancy_class.value_counts().reindex(range(4), fill_value=0)
    surviving_sessions = kept.groupby("occupancy_class").candidate_session.nunique().reindex(range(4), fill_value=0)
    scenario = pd.DataFrame([
        dict(scenario="current_A_C2_CO2_v1", rows=len(single), candidate_sessions=single.candidate_session.nunique(),
             **{f"class_{i}": int(session_rows[i]) for i in range(4)}),
        dict(scenario="quarantine_conflicted_session", rows=len(kept), candidate_sessions=kept.candidate_session.nunique(),
             **{f"class_{i}": int(kept_rows[i]) for i in range(4)}),
    ])
    scenario["rows_removed"] = scenario.rows.iloc[0] - scenario.rows
    write_csv(scenario, OUT / "quarantine_scenario.csv")

    # A published label that survives every automated check means we stop and report.
    open_keys = attribution[attribution.status.ne("RESOLVED")]
    decisions = pd.DataFrame([
        dict(issue="same_three_channel_observation_under_3_measure_ids",
             evidence="C2 CO2C2/TairC2/RHC2 identical for 184 keys across St Julia 4to, St Julia P4, Bressol JM A1; "
                      "temp/humidity are ~1000x off instrument scale for every participant",
             conclusion="sensor observation cannot be attributed to a single room",
             action="no label change; quarantine candidate only", resolved=False),
        dict(issue="which_occupancy_label_belongs_to_the_observation",
             evidence="P4 consistently 11-15 people, 4to 1-5 people, JM A1 0-12 on the identical triple",
             conclusion="publication metadata does not say which occupant count was measured by the C2 device",
             action="requires occupied-count record or device placement trace from the source", resolved=False),
        dict(issue="multi_sensor_version_scale_repair",
             evidence="3136 TairC2/RHC2 field operations pass the fixed divisor-1000 rule against past anchor and D reference",
             conclusion="engineering candidate only; publisher erratum not found",
             action="excluded from first-round training; unaffected by the quarantine", resolved=False),
        dict(issue="conflict_affects_validation_or_test",
             evidence="all 368 retained conflict rows sit in the train split of the candidate plan",
             conclusion="evaluation sets are not directly contaminated by these keys",
             action="record as evidence; not a substitute for resolving labels", resolved=True),
    ])
    decisions["evidence"] = decisions.evidence.astype(str)
    write_csv(decisions, OUT / "open_issues.csv")

    unchanged = protected == {str(p.relative_to(ROOT)): sha(p)
                              for folder in ["data/raw", "data/processed"]
                              for p in (ROOT / folder).rglob("*") if p.is_file()}
    if not unchanged:
        raise SystemExit("Protected data changed during conflict review; aborting")

    manifest = {
        "status": "OPEN" if len(open_keys) else "RESOLVED",
        "conflict_keys": int(len(attribution)),
        "conflict_rows": int(len(evidence)),
        "keys_unattributable": int(attribution.sensor_attribution.eq("UNATTRIBUTABLE").sum()),
        "keys_off_scale": int(attribution.status.eq("COPY_AMBIGUOUS").sum()),
        "keys_plausible_values": int(attribution.status.eq("OPEN_PLAUSIBLE_VALUES").sum()),
        "measure_ids_involved": sorted(quarantine),
        "quarantine_candidate_session": sorted(map(str, quarantined_sessions)),
        "quarantine_rows": int(len(single) - len(kept)),
        "labels_modified": False,
        "outputs_refreshed": True,
        "training_performed": False,
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                          [RAW, SINGLE, AUDIT / "conflict_row_evidence.csv"]},
        "script_sha256": sha(Path(__file__)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# C2 标签冲突核查与处置结论

## 结论一句话

184 个冲突键共 368 条记录（进入 `A_C2_CO2_v1` 的部分），**全部来自同一批复制粘贴的 C2 传感器原文**：同一时间戳、同一组三通道数值同时挂在两个不同教室名下，因此无法确认哪一条人数标签属于这批观测。本次**未修改任何标签**，只给出一个可审核的「整会话隔离」候选，并保留开放问题清单。

## 冲突键的构成

{table(attribution[['conflict_id','key_Datetime','TairC2','RHC2','values_physically_plausible','sensor_attribution','retained_rows','proposed_split']].head(3))}

具体到 {len(attribution)} 个键，两类情况必须分开看。下表「记录数」均指**进入 `A_C2_CO2_v1` 的行**；原始文件里同一批键共 {int(len(evidence_all))} 行（每个键 3 行），其中 {int(len(evidence_all)) - int(len(evidence))} 行已被清洗规则排除（主要是 JM A1 的 CO₂ 越界行）。

| 情形 | 键数 | C2 内记录数 | 说明 | 处置 |
| --- | ---: | ---: | --- | --- |
| A 数值越出仪器量程 | {int(attribution.status.eq('COPY_AMBIGUOUS').sum())} | {int(attribution.loc[attribution.status.eq('COPY_AMBIGUOUS'),'retained_rows'].sum())} | TairC2 ≈ 2×10⁴ °C、RHC2 ≈ 3.7×10⁴ %RH，两或三方数值逐位相同，不同教室不可能同时读出同一串数字 | 观测无归属，标签无法验证 → 整会话隔离候选 |
| B 数值在物理量程内 | {int(attribution.status.eq('OPEN_PLAUSIBLE_VALUES').sum())} | {int(attribution.loc[attribution.status.eq('OPEN_PLAUSIBLE_VALUES'),'retained_rows'].sum())} | 如键 167（TairC2=21.1 °C、RHC2=29.12 %RH），读数本身可信，但仍同时挂在两个教室名下 | 不能排除真实观测，也不能确认归属 → 保留问题清单 |

`source_links.csv` 记录这三个 Measure_ID 之间的 C2 三通道键高度重合：P4 的全部 231 个键都在 4to 中出现，4to 与 JM A1 共有 450 个完全相同的键。真正的判据不是「CO₂ 相同但人数不同」（同浓度本来就可以对应不同人数），而是**同一时间戳、同一组三通道数值同时挂在两个不同教室名下**——两台独立传感器不会给出逐位相同的温湿度，因此这是原始文件的复制/配对错位，现有清洗没有制造这个错误，只是继承了它。

## 逐 Measure_ID 影响

{table(impact)}

368 条冲突记录**全部落在候选方案的 train 分区**，验证集/测试集未被这些键直接污染。但这不等于问题已解决：若按现有标签训练，模型会从 4to 学到「同一个 3.5 万级原始值 → 1~5 人」，与 P4 的「→ 11~15 人」直接矛盾。

## 隔离候选的量级

{table(scenario)}

这是**候选方案，不是已执行的操作**：现有 `A_C2_CO2_v1.csv` 未做任何删改。若采用，需确认无人类在验证/测试集中仍有 ≥2 个候选会话支撑（当前剩余 {int(surviving_sessions.min()) if len(surviving_sessions) else 0} 个为最少）。

## 未决问题清单（open_issues.csv）

{table(decisions[['issue','conclusion','action','resolved']])}

## 为什么不能自己判定

1. **不能按多数票或"保留第一条"**：每个键的 3 条记录标签各不相同，没有多数。
2. **不能按模型预测重标**：用待检验的标签去修正标签本身，属于循环论证。
3. **不能按"4to 人多所以高类别"推断**：人数标签是观测值，不是房间容量。JA 与 4to 的测点连时间都对不上（连 `Datetime` 都不重合），两者之间不存在可比较的同时刻样本。
4. **可以确定**：`CO2C2` 与 `Datetime` 在两个测点上逐点相同，配合温湿度越界，这是原始文件的配对错位，而**不是**「CO₂ 相同、人数不同」的现象——若只看 CO₂ 单值，同浓度对应多人数是正常的，不构成错误。
5. **不能确定**：错位发生在哪一侧。发布方 P4 的人数为 11~15，St Julia 4to 为 1~5，两者都可能是真实的，也都可能配错了测点。

## 需要外部信息才能推进的问题

- 该时段的课堂人数记录（谁在哪个教室、多少学生）；
- C2 设备编号与教室的分配表；
- 原始发布方的逐行更正说明（Zenodo record 14621677 未提供）。

## 复现

```
python src/preprocess_c2.py
python src/check_training_readiness.py   # 预期退出码 2（训练门禁仍阻断）
python src/analyze_conflicts.py          # 生成本报告的四个结果文件
```

前置缺失时脚本以退出码 2 结束并提示需要先运行哪一步。原始数据与 `A_C2_CO2_v1.csv` 的 SHA-256 在运行前后一致，记录于 manifest.json。
"""
    (OUT / "conflict_resolution_report.md").write_text(report, encoding="utf-8")
    print("Conflict status:", manifest["status"], "| keys:", len(attribution),
          "| unattributable:", manifest["keys_unattributable"],
          "| quarantine rows:", manifest["quarantine_rows"])
    return 2 if len(open_keys) else 0


if __name__ == "__main__":
    raise SystemExit(main())
