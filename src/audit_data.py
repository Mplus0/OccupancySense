"""Read-only audit of OccupancySense sources; run: python src/audit_data.py.

No cleaning, feature engineering, train/test split or model fitting is performed.
Range checks are screening heuristics, not device specifications. Division by
1000 is tested only as a diagnostic hypothesis and never written back to data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "A": ("ieq-dataset.csv", "Occupancy", "Datetime", 60),
    "B": ("Classroom_data_combined_main.xlsx", "Occupants_number", "Date_Time", 31),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def table(frame):
    """Markdown without the optional tabulate dependency."""
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def rule(column):
    """Broad screening ranges with explicit assumed units; bounds inclusive."""
    low = column.lower()
    if column.startswith("DewPoint"):
        return "dew_point", -50, 50, "assumed degC"
    if column.startswith(("Tair", "Tg")) or "temperature" in low:
        return "temperature", -50, 60, "assumed degC"
    if column.startswith("RH") or "humidity" in low:
        return "humidity", 0, 100, "assumed %RH"
    if column.startswith("CO2"):
        return "co2", 250, 10000, "assumed ppm; review range, not physical bound"
    if column.startswith("PM"):
        return "particles", 0, 1000, "assumed ug/m3; review range"
    if column == "PrD":
        return "pressure", 800, 1100, "assumed hPa; review range"
    if column == "VaD" or "wind speed" in low:
        return "air_speed", 0, (20 if column == "VaD" else 200), "assumed m/s" if column == "VaD" else "km/h"
    if column == "VOCD":
        return "voc", 0, np.inf, "unit unknown; negative check only"
    if column == "Metabolic":
        return "metabolic", 0.5, 10, "assumed met; review only (zero may be sentinel)"
    if "blind" in low:
        return "blind", 0, 100, "%"
    if "state" in low:
        return "state", 0, 1, "binary 0/1"
    return None


def audit(key, raw, metrics, figures, as_of):
    filename, label, time_col, maximum = SOURCES[key]
    path = raw / filename
    before = digest(path)
    sheets = None
    if path.suffix == ".xlsx":
        with pd.ExcelFile(path) as book:
            sheets = book.sheet_names
            if len(sheets) != 1:
                raise ValueError(f"Expected one worksheet in {path}, found {sheets}; select scope explicitly")
            df = pd.read_excel(book, sheet_name=sheets[0])
    else:
        df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty dataset: {path}")
    n = len(df)
    y = pd.to_numeric(df[label], errors="coerce")
    valid_y = y.notna() & np.isfinite(y) & y.between(0, maximum) & y.mod(1).eq(0)
    times = pd.to_datetime(df[time_col], errors="coerce")
    # Microsecond timestamps avoid overflow in differences involving year 2222.
    times = times.astype("datetime64[us]")
    future = times.gt(pd.Timestamp(as_of) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
    profile = pd.DataFrame({
        "column": df.columns, "dtype": df.dtypes.astype(str).values,
        "non_missing": df.notna().sum().values, "missing": df.isna().sum().values,
        "missing_rate": df.isna().mean().values,
        "unique_non_missing": df.nunique().values,
        "blank_string_count": [int(df[c].astype("string").str.fullmatch(r"\s*").fillna(False).sum()) for c in df],
    })
    csv(profile, metrics / f"{key}_columns.csv")
    csv(df.select_dtypes("number").describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.reset_index(names="column"), metrics / f"{key}_numeric_summary.csv")
    distribution = y.value_counts(dropna=False).sort_index().rename_axis("occupants").reset_index(name="count")
    distribution["fraction_all_rows"] = distribution["count"] / n
    csv(distribution, metrics / f"{key}_occupancy_distribution.csv")

    # Evidence contains source row numbers (header is row 1), never corrected values.
    evidence = []
    def flag(mask, column, issue):
        for i in df.index[mask.fillna(False)]:
            evidence.append({"source_row": int(i) + 2, "column": column, "issue": issue,
                             "raw_value": str(df.at[i, column]), "timestamp": str(df.at[i, time_col]),
                             "Measure_ID": str(df.at[i, "Measure_ID"]) if "Measure_ID" in df else ""})

    flag(~valid_y, label, "missing_or_invalid_label")
    flag(times.isna(), time_col, "unparseable_or_missing_time")
    flag(future, time_col, "future_time")
    anomalies, scale_groups = [], []
    for col in df.select_dtypes("number"):
        x = df[col]
        bounds = rule(col)
        finite = x.notna() & np.isfinite(x)
        flag(x.notna() & ~np.isfinite(x), col, "nonfinite")
        if bounds is None:
            continue
        kind, lo, hi, unit = bounds
        outside = finite & ~x.between(lo, hi)
        if kind == "state":
            outside |= finite & ~x.isin([0, 1])
        candidate = outside & (x / 1000).between(lo, hi) & (x.abs() >= 1000)
        q1, q3 = x[finite].quantile([.25, .75])
        iqr = q3 - q1
        iqr_count = int((finite & ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr))).sum())
        anomalies.append({"column": col, "kind": kind, "assumed_unit": unit,
                          "lower": lo, "upper": hi, "observed": int(finite.sum()),
                          "min": x[finite].min(), "median": x[finite].median(), "max": x[finite].max(),
                          "outside_range": int(outside.sum()), "outside_rate_observed": float(outside.sum() / finite.sum()) if finite.any() else None,
                          "in_range_raw": int((finite & ~outside).sum()),
                          "divide_1000_candidates": int(candidate.sum()), "iqr_outliers_raw": iqr_count})
        flag(outside, col, "outside_screening_range")
        if key == "A":
            for group, indices in df.groupby("Measure_ID", dropna=False).groups.items():
                scale_groups.append({"Measure_ID": group, "column": col, "rows": len(indices),
                                     "missing": int(x.loc[indices].isna().sum()),
                                     "in_range_raw": int((finite & ~outside).loc[indices].sum()),
                                     "outside_range": int(outside.loc[indices].sum()),
                                     "divide_1000_candidates": int(candidate.loc[indices].sum())})
    anomaly_df = pd.DataFrame(anomalies)
    csv(anomaly_df, metrics / f"{key}_sensor_checks.csv")
    if scale_groups:
        csv(pd.DataFrame(scale_groups), metrics / "A_sensor_checks_by_measure.csv")

    feature_sets = ({"D_core": ["CO2D", "TairD", "RHD"],
                     "D_core_plus_outdoor": ["CO2D", "TairD", "RHD", "TairE"]}
                    if key == "A" else {"indoor_temperature_humidity": [c for c in df if c.startswith(("TemperatureF", "HumidityF"))]})
    coverage = []
    for name, columns in feature_sets.items():
        present = valid_y & df[columns].notna().all(axis=1)
        in_bounds = present.copy()
        for col in columns:
            _, lo, hi, _ = rule(col)
            in_bounds &= np.isfinite(df[col]) & df[col].between(lo, hi)
        coverage.append({"feature_set": name, "columns": "; ".join(columns),
                         "valid_label_and_present": int(present.sum()),
                         "also_within_screening_ranges": int(in_bounds.sum()),
                         "fraction_all_rows": float(in_bounds.mean())})
    csv(pd.DataFrame(coverage), metrics / f"{key}_candidate_feature_coverage.csv")

    duplicate_checks = []
    def duplicates(name, columns):
        mask = df.duplicated(columns, keep=False)
        duplicate_checks.append({"definition": name, "extra_rows": int(df.duplicated(columns).sum()),
                                 "all_involved_rows": int(mask.sum())})
        flag(mask, time_col, f"duplicate:{name}")
    duplicates("all_columns", list(df.columns))
    excluded = [c for c in ["Measure_ID", "Unnamed: 0"] if c in df]
    duplicates("excluding_identifiers", [c for c in df if c not in excluded])
    if key == "A":
        duplicates("measure_and_time", ["Measure_ID", time_col])
        # Aula is not globally unique; this is only a candidate collision check.
        duplicates("room_name_and_time_candidates", ["Aula", time_col])
    else:
        duplicates("timestamp", [time_col])
    csv(pd.DataFrame(duplicate_checks), metrics / f"{key}_duplicates.csv")

    work = df.assign(_time=times, _label=y, _valid_label=valid_y)
    group_columns = ["Measure_ID", "Aula"] if key == "A" else []
    groups = {}
    for col in group_columns:
        grouped = work.groupby(col, dropna=False)
        g = grouped.agg(rows=(label, "size"), labeled_rows=("_valid_label", "sum"),
                        label_unique=(label, "nunique"), label_min=("_label", "min"),
                        label_max=("_label", "max"), start=("_time", "min"), end=("_time", "max"))
        g["label_missing_rate"] = 1 - g.labeled_rows / g.rows
        g["zero_rows"] = grouped["_label"].agg(lambda s: s.eq(0).sum())
        if col == "Measure_ID":
            g["room_names"] = grouped.Aula.agg(lambda s: "; ".join(sorted(s.dropna().unique())))
        csv(g.reset_index(), metrics / f"{key}_groups_{col}.csv")
        csv(grouped[[c for c in df.columns if c != col]].agg(lambda s: s.isna().mean()).reset_index(), metrics / f"{key}_missing_by_{col}.csv")
        groups[col] = {"unique_non_missing": int(df[col].nunique()), "missing_rows": int(df[col].isna().sum()),
                       "constant_label_groups": int(g.label_unique.eq(1).sum()),
                       "smallest_group": int(g.rows.min()), "largest_group": int(g.rows.max())}

    mismatch = pd.Series(False, index=df.index)
    overlap_pairs = []
    if key == "A":
        id_date = pd.to_datetime(df.Measure_ID.str.extract(r"^(\d{8})", expand=False), format="%Y%m%d", errors="coerce")
        mismatch = times.notna() & id_date.notna() & times.dt.normalize().ne(id_date)
        flag(mismatch, time_col, "date_disagrees_with_measure_prefix")
        # Exact equality on ALL fields except Measure_ID, not a hash approximation.
        cols = [c for c in df if c != "Measure_ID"]
        matches = df[df.duplicated(cols, keep=False)]
        pair_counts = {}
        for _, same in matches.groupby(cols, dropna=False):
            ids = sorted(same.Measure_ID.dropna().unique())
            for i, left in enumerate(ids):
                for right in ids[i + 1:]:
                    pair_counts[(left, right)] = pair_counts.get((left, right), 0) + 1
        overlap_pairs = [{"measure_left": a, "measure_right": b, "identical_observations": count}
                         for (a, b), count in sorted(pair_counts.items())]
        csv(pd.DataFrame(overlap_pairs, columns=["measure_left", "measure_right", "identical_observations"]), metrics / "A_cross_measure_duplicates.csv")

    intervals = []
    iter_groups = work.groupby("Measure_ID", dropna=False) if key == "A" else [("all", work)]
    for name, part in iter_groups:
        original = part._time.dropna()
        ordered = part.sort_values("_time")
        delta = ordered._time.diff().dt.total_seconds()
        good_pairs = delta.gt(0) & ordered._valid_label & ordered._valid_label.shift(fill_value=False)
        same_label = ordered._label.eq(ordered._label.shift()) & good_pairs
        positive = delta[delta.gt(0)]
        intervals.append({"group": name, "rows": len(part), "backward_steps_source_order": int(original.diff().dt.total_seconds().lt(0).sum()),
                          "zero_intervals": int(delta.eq(0).sum()), "median_positive_seconds": positive.median(),
                          "max_positive_seconds": positive.max(), "valid_adjacent_label_pairs": int(good_pairs.sum()),
                          "same_adjacent_label_pairs": int(same_label.sum()),
                          "label_lag1_correlation": ordered.loc[good_pairs, "_label"].corr(ordered._label.shift().loc[good_pairs]) if good_pairs.sum() > 2 and ordered.loc[good_pairs, "_label"].nunique() > 1 and ordered._label.shift().loc[good_pairs].nunique() > 1 else None})
    interval_df = pd.DataFrame(intervals)
    csv(interval_df, metrics / f"{key}_time_intervals.csv")
    daily = work.groupby(work._time.dt.date).agg(rows=(label, "size"), labeled_rows=("_valid_label", "sum"), mean_occupancy=("_label", "mean"))
    csv(daily.reset_index(names="date"), metrics / f"{key}_daily_summary.csv")
    csv(pd.DataFrame(evidence, columns=["source_row", "column", "issue", "raw_value", "timestamp", "Measure_ID"]), metrics / f"{key}_issue_rows.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    counts = y[valid_y].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color="#287d91")
    axes[0].set(xlabel="Occupants", ylabel="Rows", title=f"{key}: occupancy (valid labels only)")
    axes[1].bar(counts.index, counts.values, color="#287d91")
    axes[1].set(yscale="log", xlabel="Occupants", ylabel="Rows (log scale)", title=f"Missing / invalid labels: {int((~valid_y).sum()):,}")
    fig.savefig(figures / f"{key}_occupancy_distribution.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, max(6, .26 * len(df.columns))), layout="constrained")
    missing = df.isna().mean().sort_values()
    ax.barh(missing.index, missing.values * 100, color="#b75a47")
    ax.set(xlim=(0, 100), xlabel="Missing cells (%)", title=f"{key}: missingness by column")
    if missing.eq(0).all():
        ax.text(.5, .5, "All columns: 0% missing", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="#287d91")
    fig.savefig(figures / f"{key}_missing_rates.png", dpi=160)
    plt.close(fig)
    after = digest(path)
    if before != after:
        raise RuntimeError(f"Source changed during audit: {path}")
    finite_time = times[times.notna() & ~future]
    pairs = int(interval_df.valid_adjacent_label_pairs.sum())
    summary = {"dataset": key, "source": str(path.resolve()), "sha256_before": before, "sha256_after": after,
               "source_unchanged": before == after, "sheet_names": sheets, "rows": n, "columns": len(df.columns),
               "missing_cells": int(df.isna().sum().sum()), "missing_cell_rate": float(df.isna().mean().mean()),
               "complete_rows_all_columns": int(df.notna().all(axis=1).sum()),
               "label_missing": int(df[label].isna().sum()), "invalid_nonmissing_labels": int((df[label].notna() & ~valid_y).sum()),
               "valid_labels": int(valid_y.sum()), "label_min": float(y[valid_y].min()), "label_max": float(y[valid_y].max()),
               "label_distinct": int(y[valid_y].nunique()), "zero_labels": int(y[valid_y].eq(0).sum()),
               "zero_fraction_labeled": float(y[valid_y].eq(0).mean()),
               "time_min_raw": str(times.min()), "time_max_raw": str(times.max()),
               "time_min_nonfuture": str(finite_time.min()), "time_max_nonfuture": str(finite_time.max()),
               "time_unparseable_or_missing": int(times.isna().sum()), "future_time_rows": int(future.sum()),
               "measure_date_mismatch_rows": int(mismatch.sum()), "groups": groups, "duplicates": duplicate_checks,
               "adjacent_equal_label_fraction": float(interval_df.same_adjacent_label_pairs.sum() / pairs) if pairs else None,
               "cross_measure_duplicate_pairs": overlap_pairs, "candidate_feature_coverage": coverage,
               "sensor_outside_counts": dict(zip(anomaly_df.column, anomaly_df.outside_range.astype(int))),
               "sensor_divide_1000_candidates": dict(zip(anomaly_df.column, anomaly_df.divide_1000_candidates.astype(int)))}
    return summary, df, anomaly_df


def report(summaries, frames, checks, output, as_of):
    a, b = summaries["A"], summaries["B"]
    comparison = []
    for s in [a, b]:
        comparison.append({"数据集": s["dataset"], "样本/字段": f'{s["rows"]:,} / {s["columns"]}',
                           "有效人数标签": s["valid_labels"], "标签缺失": s["label_missing"],
                           "单元格缺失率": f'{s["missing_cell_rate"]:.2%}',
                           "空教室占有效标签": f'{s["zero_fraction_labeled"]:.2%}',
                           "人数范围/取值数": f'{s["label_min"]:g}–{s["label_max"]:g} / {s["label_distinct"]}',
                           "整行重复(多余行)": s["duplicates"][0]["extra_rows"]})
    comparison = pd.DataFrame(comparison)
    csv(comparison, output / "metrics" / "audit_comparison.csv")
    suspect = checks["A"].query("divide_1000_candidates > 0")[["column", "observed", "in_range_raw", "outside_range", "divide_1000_candidates"]]
    b_issues = checks["B"].query("outside_range > 0")[["column", "min", "max", "outside_range"]]
    text = f"""# OccupancySense 数据集质量审核

审核日期：{as_of}。只读审核；未清洗、插补、修正量纲、构造训练特征、划分训练集或训练模型；未合并 A/B。两份输入的运行前后 SHA-256 一致，见 `metrics/audit_summary.json`。

## 推荐结论

**建议将 B 作为当前工程基线主数据集，A 作为修复与溯源后的多教室扩展数据集。** B 标签完整、时间规则、室内温湿度量纲一致，更适合先建立可复现流程；但 90% 为空教室、人数只有少量离散取值、没有 CO₂ 或教室标识，不能据此证明精细人数估计或跨教室泛化能力。若项目目标必须包含 CO₂ 或跨教室人数估计，则保留 A 为目标主数据集，但在解决量纲、重复来源与标签缺失前不宜直接训练。

## 数据概况

{table(comparison)}

有效标签要求：可解析为有限数值、非负整数，且 A ≤ 60、B ≤ 31（本项目已知范围，不是普遍人数上限）。A/B 的非缺失非法标签分别为 {a['invalid_nonmissing_labels']} / {b['invalid_nonmissing_labels']}。缺失率分母为全部行；人数分布同时提供全样本占比，空教室占比分母为有效标签。A 全字段完整行仅 {a['complete_rows_all_columns']} 条，B 为 {b['complete_rows_all_columns']} 条，不建议对 A 全字段直接 dropna。

## 传感器与量纲

阈值为审核筛查假设，不是传感器量程或认证标准：温度 −50～60 °C、露点 −50～50 °C、湿度 0～100%、CO₂ 250～10000 ppm、颗粒物 0～1000（单位暂假设 µg/m³）、气压 800～1100 hPa、室内风速 0～20 m/s、室外风速 0～200 km/h。VOC 仅查非负；Metabolic 暂按 0.5～10 met 筛查。边界包含端点，空值不计入传感器异常分母。离群值仅标记，不能等同错误；原始尺度 1.5×IQR 结果另存，不用于自动删除。

A 原始 CSV 已含 `19025.0`、`41742.0` 等值，问题并非本次读入后产生。以下列同时存在正常量级和异常量级；“÷1000 候选”仅表示原值越界、绝对值 ≥1000 且除以 1000 后落入筛查区间：

{table(suspect)}

这支持部分行存在千倍尺度或小数解析错误的**假设**，不证明所有异常都应除以 1000，尤其 CO₂、PM 和 Metabolic 需要原始说明与设备量程核实。不能整列统一除以 1000。按 Measure_ID 的异常/缺失统计见 `A_sensor_checks_by_measure.csv`，可判断组内是否也混用尺度。CO2D、TairD、TgD、RHD 等可作为人工交叉核验参考，但不同设备位置与响应时间未必相同。

B 的筛查异常：

{table(b_issues)}

室外湿度略超 100%、风速为负，应核查天气数据插值或采集来源，不能凭数值外观断定原因。设定温度 0 可能表示关闭/不适用，需结合 HVAC 状态解释，不能自动视为真实 0 °C。B 温度列名实际含 `°C`；若终端显示乱码，应检查终端编码，不应据此修改原始列名。

## 时间、分组和重复风险

- A 原始时间范围：{a['time_min_raw']} 至 {a['time_max_raw']}。有 {a['future_time_rows']} 条未来时间，排除未来时间后的诊断范围为 {a['time_min_nonfuture']} 至 {a['time_max_nonfuture']}；这只是统计过滤，未修改数据。有 {a['measure_date_mismatch_rows']} 行日期与 Measure_ID 的前八位日期不同，不能未经核对就按 ID 覆盖。
- B 时间范围：{b['time_min_raw']} 至 {b['time_max_raw']}。本次检查相邻间隔均为 600 秒（10 分钟），无倒序或重复时间，见 `B_time_intervals.csv`。A/B 时间解析失败或缺失分别为 {a['time_unparseable_or_missing']} / {b['time_unparseable_or_missing']}。
- A 有 {a['groups']['Measure_ID']['unique_non_missing']} 个 Measure_ID、{a['groups']['Aula']['unique_non_missing']} 个非空 Aula 名称；Aula 缺失 {a['groups']['Aula']['missing_rows']} 行。Aula 是名称而非全局教室 ID，如 A1/A2 可能在不同学校复用，不能把名称数直接当实际教室数。学校信息嵌于 Measure_ID，但命名格式与别名不统一，需来源表确认。
- 两份数据全字段重复均为 0；A 排除 Measure_ID 后有 **{a['duplicates'][1]['extra_rows']} 条多余重复行**，跨 ID 的相同观测详见 `A_cross_measure_duplicates.csv`。仅用原始 Measure_ID 分组仍可能泄漏；应先确认同源测量与别名，并把相关组放入同一分区。
- A 的 Aula+Datetime 候选重复多余行数为 {a['duplicates'][3]['extra_rows']}，可能含不同学校同名教室、缺失 Aula 或同一观测重复，不能直接据此删除。Measure_ID+时间重复多余行数为 {a['duplicates'][2]['extra_rows']}。
- 按测量组内时间排序，相邻有效标签相同占比 A 为 {a['adjacent_equal_label_fraction']:.2%}，B 为 {b['adjacent_equal_label_fraction']:.2%}（仅正时间间隔；未排除长间隔）。这是序列依赖的诊断，不能视为独立样本证据。随机按行拆分容易把相邻时刻、同一课堂、同源重复分到两边。
- A 应按确认后的学校+物理教室/测量会话分组评估；跨教室目标应留出整间教室或学校。同一教室未来预测采用时间外推。B 缺少教室及 Measure_ID 字段，只能确认一条时间序列，无法从本文件证实教室数量；建议按连续日期/周留出后期测试集，窗口跨边界时设置隔离带，长度取决于后续窗口和预测跨度。
- 所有未来的插补、标准化、特征选择及重采样都只能在训练分区拟合。标签、同源 ID、时间索引不直接作为环境传感器输入。日期/时段可能记住课表，若后续使用应单独做消融，并核实推理时可获取。

## 可用特征与后续建议

| 数据集 | 候选特征 | 条件与暂缓项 |
| --- | --- | --- |
| B | 六组室内 TemperatureF*/HumidityF*；室外温度 | 核心候选；六个通道不能当成六间教室。验证传感器冗余，保持时间留出 |
| B | 室外湿度、风速 | 先核实异常的来源和处理规则 |
| B | 门窗、百叶、HVAC、Set point temperature | 仅在预测当时可获取时使用；控制动作可能响应人数，需要单独评估，防止标签生成或未来控制信息泄漏 |
| A | CO2D、TairD、TgD、RHD、TairE、PrD、VaD、VOCD | 原始尺度相对可解释，但缺失与设备/会话相关，需核实单位、按所选特征统计有效覆盖 |
| A | TairC1/C2、RHC1/C2、DewPointC1/C2、CO2C1/C2、CO2R、TairR、RHR、PM* | 量纲与采集来源未解决前暂缓 |
| A | Ventilation、Distribution | 需确认是否部署时可获得的记录，以及是否与人数标注相互派生 |

不将 Occupancy/Occupants_number 作为输入；B `Unnamed: 0` 排除，A `Measure_ID`/`Aula` 仅用于分组审计，`C1/C2/D/R` 含义未确认且缺失严重，`Metabolic` 来源及异常未确认，均不推荐直接入模。标签缺失不当作 0、不用全量数据推断标签。

候选特征与有效标签的共同覆盖（仅计数，未筛选导出数据，也未处理重复）：

{table(pd.DataFrame([dict(数据集=k, 特征组=c['feature_set'], 标签与特征均有效且通过范围检查=c['also_within_screening_ranges']) for k, s in summaries.items() for c in s['candidate_feature_coverage']]))}

后续顺序：① 取回单位、设备、人数标注及天气处理说明，核对 A 的千倍尺度、2222 年份和测量别名；② 制定逐字段、逐会话且可追溯的清洗规则，确认重复观测后再处理；③ 选定任务与独立分组/时间留出方案；④ 只在训练数据上执行清洗参数估计及后续特征工程。此次仅给建议，没有执行这些步骤。

B 的零人数基线在“有人/无人”任务上即可达到 {b['zero_fraction_labeled']:.2%} 的表面准确率。后续应同时报告有人召回率、平衡指标以及人数 MAE/RMSE，分空/非空、人数区间报告误差；B 只覆盖 {b['label_distinct']} 种人数值，A 覆盖 {a['label_distinct']} 种，需检查留出分区中的标签覆盖，不能把未出现人数的泛化当作已验证。

## 产物与复现

运行 `python src/audit_data.py`；可用 `--as-of {as_of}` 固定未来时间判定日期。默认路径相对于脚本所在项目根目录，因此可从任意工作目录执行。依赖复用 pandas、numpy、matplotlib、openpyxl，无新增依赖。

- `metrics/audit_summary.json`：机器可读摘要、阈值说明、输入哈希和版本。
- `metrics/A_*`、`metrics/B_*`：字段/缺失率、数值分位数、人数分布、传感器检查、重复定义、分组、日统计和时间间隔；`*_issue_rows.csv` 为完整问题定位清单，源文件行号含表头（首条数据=2），同一行可能出现多个问题，问题条目数不能当独立异常行数。
- `figures/*_occupancy_distribution.png`、`*_missing_rates.png`、`A_sensor_scales.png`、`B_daily_occupancy.png`：人数、缺失和重点诊断图。全部为原始数据统计，未导出清洗数据。

![A 人数分布](figures/A_occupancy_distribution.png)
![B 人数分布](figures/B_occupancy_distribution.png)
![A 缺失率](figures/A_missing_rates.png)
![B 缺失率](figures/B_missing_rates.png)
![A 原始尺度](figures/A_sensor_scales.png)
![B 每日人数](figures/B_daily_occupancy.png)
"""
    (output / "data_audit_report.md").write_text(text, encoding="utf-8")


def diagnostic_plots(frames, figures):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, col in zip(axes.flat, ["TairC1", "RHC1", "DewPointC1", "CO2C1", "TairR", "CO2R"]):
        x = frames["A"][col].dropna()
        positive = x[x > 0]
        ax.hist(positive, bins=np.geomspace(positive.min(), positive.max(), 45), color="#287d91")
        ax.set(xscale="log", xlabel="Raw value (log scale)", ylabel="Rows", title=col)
    fig.suptitle("A: raw sensor magnitudes (no rescaling)")
    fig.savefig(figures / "A_sensor_scales.png", dpi=160)
    plt.close(fig)
    b = frames["B"]
    daily = b.groupby(b.Date_Time.dt.date).Occupants_number.agg(["mean", "max"])
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(daily.index, daily["max"], label="Daily maximum", color="#b75a47")
    ax.plot(daily.index, daily["mean"], label="Daily mean", color="#287d91")
    ax.set(ylabel="Occupants", title="B: daily occupancy (original labels)")
    ax.legend()
    fig.savefig(figures / "B_daily_occupancy.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    raw, output = args.raw_dir.resolve(), args.output_dir.resolve()
    if output == raw or raw in output.parents:
        parser.error("Output directory must not be inside the raw input directory")
    metrics, figures = output / "metrics", output / "figures"
    for folder in [metrics, figures]:
        folder.mkdir(parents=True, exist_ok=True)
    summaries, frames, checks = {}, {}, {}
    for key in SOURCES:
        summaries[key], frames[key], checks[key] = audit(key, raw, metrics, figures, args.as_of)
        print(f"{key}: {summaries[key]['rows']:,} rows; {summaries[key]['valid_labels']:,} valid labels")
    diagnostic_plots(frames, figures)
    report(summaries, frames, checks, output, args.as_of)
    payload = {"as_of": str(args.as_of), "versions": {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
               "method": "Read-only; thresholds are screening assumptions, not device specifications; divide-1000 is diagnostic only. See sensor_checks CSVs for bounds; nonfinite numbers exported as null.",
               "datasets": summaries}
    # Normalize NumPy scalars and any nonfinite diagnostic values to strict JSON.
    payload = json.loads(json.dumps(payload, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
    def strict(value):
        if isinstance(value, dict):
            return {k: strict(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strict(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    (metrics / "audit_summary.json").write_text(json.dumps(strict(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report: {output / 'data_audit_report.md'}")


if __name__ == "__main__":
    main()
