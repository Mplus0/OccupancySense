"""Deterministic A/D_core v1 cleaning. No imputation, scaling, split or training.

Run from any directory: python /path/to/OccupancySense/src/preprocess.py
Original files are read-only; uncertain duplicates are retained and flagged.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import rule, table

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["CO2D", "TairD", "RHD"]
CLASS_NAMES = ["无人", "低人数", "中人数", "高人数"]
VERSION = "A_D_core_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def classes(y):
    """Caller supplies validated, nonnegative integer labels."""
    return pd.Series(np.select([y.eq(0), y.le(5), y.le(15)], [0, 1, 2], default=3), index=y.index)


def clean(df, as_of):
    """Return retained data and evidence without mutating the input frame."""
    required = ["Occupancy", "Datetime", "Aula", "Measure_ID", *FEATURES]
    missing = set(required) - set(df.columns)
    if missing or df.empty:
        raise ValueError(f"Empty input or missing columns: {sorted(missing)}")
    y = pd.to_numeric(df.Occupancy, errors="coerce")
    times = pd.to_datetime(df.Datetime, format="%Y-%m-%d %H:%M:%S", errors="coerce").astype("datetime64[us]")
    label_missing = df.Occupancy.isna() | df.Occupancy.astype("string").str.strip().eq("").fillna(False)
    valid = y.notna() & np.isfinite(y) & y.between(0, 60) & y.mod(1).eq(0)
    core = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    sensor_missing = df[FEATURES].isna().any(axis=1)
    sensor_invalid = (~np.isfinite(core)).any(axis=1) & ~sensor_missing
    outside = pd.Series(False, index=df.index)
    for col in FEATURES:
        _, lo, hi, _ = rule(col)
        outside |= np.isfinite(core[col]) & ~core[col].between(lo, hi)
    checks = [
        ("missing_label", label_missing, "人数标签缺失；不当作0或插补"),
        ("invalid_label", ~valid & ~label_missing, "人数必须为有限整数且在0~60内"),
        ("invalid_time", times.isna(), "时间缺失或不符合YYYY-MM-DD HH:MM:SS"),
        ("future_time", times.dt.date.gt(as_of), "晚于固定审核日期；隔离，不推测修正年份"),
        ("missing_D_core", sensor_missing, "D_core不完整；v1采用完整观测，不插补"),
        ("invalid_D_core", sensor_invalid, "D_core包含非数字或非有限值"),
        ("out_of_range_D_core", outside, "超出预先确定的审核范围；隔离待核实，不裁剪"),
    ]
    keep = pd.Series(True, index=df.index)
    first = pd.Series("", index=df.index)
    flags = pd.DataFrame(index=df.index)
    steps = []
    for code, mask, reason in checks:
        mask = mask.fillna(False)
        flags[code] = mask
        removed = keep & mask
        steps.append(dict(rule=code, reason=reason, matched_raw_rows=int(mask.sum()),
                          input_rows=int(keep.sum()), removed_this_step=int(removed.sum()),
                          remaining_rows=int((keep & ~mask).sum())))
        first.loc[removed] = code
        keep &= ~mask

    # Review-only flags never remove rows or overwrite original identifiers.
    review = pd.DataFrame(index=df.index)
    review["exact_full_duplicate"] = df.duplicated(keep=False)
    non_id = [c for c in df if c != "Measure_ID"]
    review["duplicate_excluding_measure"] = df.duplicated(non_id, keep=False)
    review["same_measure_timestamp"] = df.duplicated(["Measure_ID", "Datetime"], keep=False)
    review["same_room_name_timestamp_candidate"] = df.duplicated(["Aula", "Datetime"], keep=False) & df.Aula.notna()
    core_key = ["Datetime", *FEATURES]
    review["same_time_D_core_candidate"] = df.duplicated(core_key, keep=False) & core.notna().all(axis=1) & times.notna()
    id_date = pd.to_datetime(df.Measure_ID.astype("string").str.extract(r"^(\d{8})", expand=False), format="%Y%m%d", errors="coerce")
    review["date_measure_mismatch"] = times.notna() & id_date.notna() & times.dt.normalize().ne(id_date)
    review["missing_or_unparsed_measure"] = df.Measure_ID.isna() | id_date.isna()
    review["missing_room"] = df.Aula.isna() | df.Aula.astype("string").str.strip().eq("").fillna(False)

    # Candidate dependency components are conservative split constraints only,
    # not assertions that the connected Measure_IDs are the same physical room.
    edges = []
    for kind, columns, mask in [
        ("identical_except_measure", non_id, review.duplicate_excluding_measure),
        ("same_time_D_core", core_key, review.same_time_D_core_candidate),
    ]:
        counts = {}
        for _, block in df.loc[mask].groupby(columns, dropna=False, sort=True):
            for left, right in combinations(sorted(block.Measure_ID.dropna().unique()), 2):
                counts[left, right] = counts.get((left, right), 0) + 1
        edges.extend(dict(left=a, right=b, evidence=kind, matching_keys=n, status="candidate_not_confirmed")
                     for (a, b), n in sorted(counts.items()))
    ids = sorted(df.Measure_ID.dropna().unique())
    parent = {x: x for x in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for edge in edges:
        a, b = find(edge["left"]), find(edge["right"])
        parent[max(a, b)] = min(a, b)
    components = {}
    for value in ids:
        components.setdefault(find(value), []).append(value)
    mapping = {}
    for members in components.values():
        group_id = "candidate_" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        for member in members:
            mapping[member] = group_id
    linked = {m for members in components.values() if len(members) > 1 for m in members}
    review["linked_measure_candidate"] = df.Measure_ID.isin(linked)
    review_reasons = review.apply(lambda row: ";".join(row.index[row]), axis=1)
    decisions = pd.DataFrame({"source_row": np.arange(len(df)) + 2, "retained": keep,
                              "exclusion_reason": first, "review_required": review.any(axis=1),
                              "review_reasons": review_reasons})
    decisions = pd.concat([decisions, flags], axis=1)
    cleaned = df.loc[keep, ["Datetime", "Aula", "Measure_ID", "Occupancy", *FEATURES]].copy()
    cleaned.insert(0, "source_row", decisions.loc[keep, "source_row"])
    cleaned["Occupancy"] = y.loc[keep].astype(int)
    cleaned["occupancy_class"] = classes(y.loc[keep])
    cleaned["occupancy_class_name"] = cleaned.occupancy_class.map(dict(enumerate(CLASS_NAMES)))
    cleaned["candidate_leakage_group"] = cleaned.Measure_ID.map(mapping)
    cleaned["review_required"] = review.any(axis=1).loc[keep]
    cleaned["review_reasons"] = review_reasons.loc[keep]
    return cleaned, decisions, pd.DataFrame(steps), review, pd.DataFrame(edges, columns=["left", "right", "evidence", "matching_keys", "status"]), mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 10, 26), help="Fixed v1 cutoff, inclusive (default: 2026-10-26)")
    args = parser.parse_args()
    source = ROOT / "data/raw/ieq-dataset.csv"
    processed = ROOT / "data/processed"
    metrics = ROOT / "results/metrics"
    # Resolve paths before any write, including Windows junction/symlink targets.
    raw = (ROOT / "data/raw").resolve()
    for folder in [processed, metrics]:
        resolved = folder.resolve()
        if resolved == raw or raw in resolved.parents:
            raise ValueError("Output cannot be inside data/raw")
        folder.mkdir(parents=True, exist_ok=True)
    original_hash = sha(source)
    df = pd.read_csv(source)
    out, decisions, steps, review, edges, mapping = clean(df, args.as_of)
    output = processed / "A_D_core_v1.csv"
    write_csv(out, output)
    write_csv(decisions, metrics / "preprocess_A_row_decisions.csv")
    write_csv(steps, metrics / "preprocess_A_processing_log.csv")
    evidence = pd.concat([decisions, df], axis=1)
    write_csv(evidence.loc[~decisions.retained], metrics / "preprocess_A_excluded_rows.csv")
    write_csv(evidence.loc[decisions.review_required], metrics / "preprocess_A_manual_review.csv")
    write_csv(edges, metrics / "preprocess_A_measure_links.csv")
    group_map = pd.DataFrame({"Measure_ID": list(mapping), "candidate_leakage_group": list(mapping.values())})
    group_map["raw_rows"] = group_map.Measure_ID.map(df.Measure_ID.value_counts()).fillna(0).astype(int)
    group_map["retained_rows"] = group_map.Measure_ID.map(out.Measure_ID.value_counts()).fillna(0).astype(int)
    write_csv(group_map, metrics / "preprocess_A_group_mapping.csv")
    counts = pd.crosstab(out.Measure_ID, out.occupancy_class).reindex(index=group_map.Measure_ID, columns=range(4), fill_value=0).fillna(0).astype(int)
    counts.columns = [f"class_{i}" for i in range(4)]
    counts["total"] = counts.sum(axis=1)
    write_csv(counts.reset_index(), metrics / "preprocess_A_group_class_counts.csv")
    raw_y = pd.to_numeric(df.Occupancy, errors="coerce")
    valid = raw_y.notna() & np.isfinite(raw_y) & raw_y.between(0, 60) & raw_y.mod(1).eq(0)
    class_counts = pd.DataFrame({"class_id": range(4), "class_name": CLASS_NAMES,
                                 "before_valid_labels": classes(raw_y[valid]).value_counts().reindex(range(4), fill_value=0).values,
                                 "after": out.occupancy_class.value_counts().reindex(range(4), fill_value=0).values})
    class_counts["after_fraction"] = class_counts.after / len(out) if len(out) else 0.
    write_csv(class_counts, metrics / "preprocess_A_class_counts.csv")
    comparison = pd.DataFrame([
        {"metric": "rows", "before": len(df), "after": len(out)},
        {"metric": "valid_labels", "before": int(valid.sum()), "after": len(out)},
        {"metric": "measure_ids", "before": df.Measure_ID.nunique(), "after": out.Measure_ID.nunique()},
        {"metric": "room_names_not_global_ids", "before": df.Aula.nunique(), "after": out.Aula.nunique()},
        {"metric": "manual_review_rows", "before": int(review.any(axis=1).sum()), "after": int(out.review_required.sum())},
        *[{"metric": f"missing_{c}", "before": int(df[c].isna().sum()), "after": int(out[c].isna().sum())} for c in FEATURES],
    ])
    write_csv(comparison, metrics / "preprocess_A_before_after.csv")

    # Diagnostic candidates only: do not write transformed sensor values anywhere.
    candidates, rules = [], []
    for col in df:
        bounds = rule(col)
        if bounds is None or col in FEATURES:
            continue
        _, lo, hi, unit = bounds
        x = pd.to_numeric(df[col], errors="coerce")
        mask = np.isfinite(x) & ~x.between(lo, hi) & x.abs().ge(1000) & (x / 1000).between(lo, hi)
        if not mask.any():
            continue
        rules.append(dict(column=col, lower=lo, upper=hi, assumed_unit=unit,
                          candidate_rows=int(mask.sum()), retained_candidate_rows=int((mask & decisions.retained).sum()),
                          rule="finite, outside bounds, abs(x)>=1000, x/1000 within bounds", action="NOT_APPLIED; verify source units and per-session mixed scales first"))
        for idx in df.index[mask]:
            candidates.append(dict(source_row=int(idx) + 2, column=col, raw_value=df.at[idx, col], Measure_ID=df.at[idx, "Measure_ID"], retained=bool(decisions.at[idx, "retained"])))
    write_csv(pd.DataFrame(rules), metrics / "preprocess_A_scale_candidate_rules.csv")
    write_csv(pd.DataFrame(candidates), metrics / "preprocess_A_scale_candidate_rows.csv")
    if sha(source) != original_hash:
        raise RuntimeError("Raw source changed during processing")
    manifest = {
        "version": VERSION, "as_of": str(args.as_of), "source": "data/raw/ieq-dataset.csv",
        "source_sha256": original_hash, "source_unchanged": True, "script_sha256": sha(Path(__file__)),
        "audit_module_sha256": sha(ROOT / "src/audit_data.py"),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__},
        "rows_before": len(df), "rows_after": len(out), "rows_excluded": len(df) - len(out),
        "default_features": FEATURES, "target": "occupancy_class", "regression_target": "Occupancy",
        "class_mapping": {"0": "无人:0", "1": "低人数:1-5", "2": "中人数:6-15", "3": "高人数:16-60"},
        "metadata_not_features": [c for c in out if c not in FEATURES + ["Occupancy", "occupancy_class", "occupancy_class_name"]],
        "bounds_inclusive": {c: list(rule(c)[1:3]) for c in FEATURES},
        "time_policy": "Strict timestamp format; exclude unparseable and dates after as_of. Retain ID-date mismatches for review.",
        "duplicate_policy": "Retain all uncertain duplicates, including identical rows. No automatic deletion, alias normalization or label correction.",
        "candidate_group_policy": "Connected components of raw cross-ID exact duplicates or identical timestamp+D_core. Conservative constraints, NOT confirmed same-source identities or final splits; no class-based grouping.",
        "review_rows_retained": int(out.review_required.sum()), "measure_groups_retained": int(out.Measure_ID.nunique()),
        "imputation": None, "scaling": None, "split": None, "training": False, "dataset_B_used": False,
        "output_sha256": sha(output),
    }
    (metrics / "preprocess_A_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = f"""# A / D_core 第一版清洗记录

运行：`python src/preprocess.py --as-of {args.as_of}`。固定规则与输入生成确定性结果，保留原始行顺序；source_row 为源文件记录号（含表头，首条数据=2）。原始文件 SHA-256 运行前后一致。B 未读取、未合并。

输入 {len(df):,} 行，输出 **{len(out):,} 行**，排除 {len(df)-len(out):,} 行；保留 {out.Measure_ID.nunique()} 个测量组。输出 `data/processed/A_D_core_v1.csv`。

## 处理数量

{table(steps)}

matched_raw_rows 可重叠；removed_this_step 按表中顺序互斥，之和等于总排除数。每行所有匹配规则及第一排除原因见 `preprocess_A_row_decisions.csv`，排除行的原始全部字段见 `preprocess_A_excluded_rows.csv`。

## 候选类别

{table(class_counts)}

分类阈值由任务指定，不从数据学习。before_valid_labels 仅统计原始合法标签。组×类别数量含清洗后为0的原始组，见 `preprocess_A_group_class_counts.csv`。

## 使用约定与遗留问题

- 默认输入严格限定为 `CO2D, TairD, RHD`；范围沿用审核报告（250~10000 ppm、−50~60 °C、0~100%），属于筛查假设，边界包含端点。未依据全量分位数删除离群值，未插补、标准化或修复量纲。
- 人数、候选分类及其中文名称均为目标相关字段，不作为输入。source_row、Datetime、Aula、Measure_ID、候选依赖组和审核标记仅为追踪/分组元数据，不作为预测特征。未对缺失教室或组标识猜测填充。
- 保留输出中的 {int(out.review_required.sum())} 条待核查记录；它们符合标签、时间和D_core基本条件，但不代表已确认独立。完整人工清单见 `preprocess_A_manual_review.csv`（包括被其他规则排除的记录，并标记 retained）。同名教室+时间碰撞只是候选；不同学校可能同名。
- `preprocess_A_measure_links.csv` 和 `preprocess_A_group_mapping.csv` 记录疑似同源依赖。candidate_leakage_group 为保守连通组，只供后续避免已知重叠跨分区，不证明实际同源，也不能替代人工确认学校/物理教室。未合并原始 Measure_ID，未删除任何疑似重复，未执行最终划分。即使相关组同分区，重复样本仍可能影响训练权重及评估，必须核查后确定处理。
- 日期与ID前缀不一致但时间本身合法的记录保留审核，不擅自改日期；明显未来时间隔离。非D_core传感器不进入此版训练输入，其原值仍在原始文件，可通过 source_row 追溯。千倍候选逐行清单与规则见 `preprocess_A_scale_candidate_*.csv`，全部标记未应用。
- 完整观测筛选改变了设备、教室和类别覆盖，不能把本版当作原始A的无偏代表；见前后对比及组统计。后续先核查重叠组和时间冲突，再确定分组/时间划分；需要拟合的处理仅用训练分区。此次未训练模型。

可复现清单、特征白名单、输入/输出/脚本哈希和实际依赖版本见 `preprocess_A_manifest.json`。
"""
    (metrics / "preprocess_A_report.md").write_text(report, encoding="utf-8")
    assert len(out) + int(steps.removed_this_step.sum()) == len(df)
    assert not out[FEATURES + ["Occupancy", "Datetime"]].isna().any().any()
    print(f"Saved {len(out):,} rows, {out.Measure_ID.nunique()} Measure_ID groups -> {output}")
    print(class_counts.to_string(index=False))


if __name__ == "__main__":
    main()
