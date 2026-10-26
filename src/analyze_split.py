"""Read-only split feasibility analysis; exports proposals, never split datasets."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import milp, Bounds, LinearConstraint

from audit_data import table, rule
from preprocess import sha, write_csv, classes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/metrics/split_feasibility"
RATIOS = np.array([.70, .15, .15])
SPLITS = ["train", "validation", "test"]


def school(measure):
    value = measure.split("_")[1]
    return "Escola Gaia" if value == "Gaia" else value


def components(ids, edges):
    parent = {i: i for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges:
        x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)
    return {i: find(i) for i in ids}


def optimize(matrix, minimum_zero=1):
    """Binary whole-unit assignment, all classes present; normalized L1 objective.

    Size deviations weight 2, each class deviation weight 1. Deviations are
    fractions of total rows / total class count, not per-split denominators.
    """
    g = len(matrix)
    values = np.column_stack([matrix.sum(axis=1), matrix])
    totals = values.sum(axis=0)
    n = g * 3 + 15
    objective = np.zeros(n)
    objective[g*3:] = np.tile(np.array([2, 1, 1, 1, 1]) / totals, 3)
    rows, low, high = [], [], []
    for i in range(g):
        row = np.zeros(n); row[i*3:i*3+3] = 1
        rows.append(row); low.append(1); high.append(1)
    for s in range(3):
        for c in range(5):
            row = np.zeros(n); row[s:g*3:3] = values[:, c]
            if c > 0:
                rows.append(row.copy()); low.append(minimum_zero if c == 1 else 1); high.append(np.inf)
            target = RATIOS[s] * totals[c]
            row[g*3+s*5+c] = -1
            rows.append(row.copy()); low.append(-np.inf); high.append(target)
            row[g*3+s*5+c] = 1
            rows.append(row.copy()); low.append(target); high.append(np.inf)
    result = milp(objective, integrality=np.r_[np.ones(g*3), np.zeros(15)],
                  bounds=Bounds(np.zeros(n), np.r_[np.ones(g*3), np.full(15, np.inf)]),
                  constraints=LinearConstraint(np.array(rows), low, high),
                  options={"time_limit": 20, "mip_rel_gap": 0.0001})
    info = {"status": int(result.status), "message": result.message,
            "objective": float(result.fun) if result.fun is not None else None,
            "mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None}
    if result.x is None:
        return None, info
    assignment = result.x[:g*3].reshape(g, 3).argmax(axis=1)
    # Never silently accept an infeasible timed-out incumbent.
    assert all((matrix[assignment == s].sum(axis=0) >= [minimum_zero, 1, 1, 1]).all() for s in range(3))
    return assignment, info


def main():
    inputs = [ROOT / "data/processed/A_D_core_v1.csv", ROOT / "data/raw/ieq-dataset.csv",
              ROOT / "results/metrics/preprocess_A_measure_links.csv"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    d = pd.read_csv(inputs[0]); raw = pd.read_csv(inputs[1])
    d["time"] = pd.to_datetime(d.Datetime)
    d["school_candidate"] = d.Measure_ID.map(school)
    OUT.mkdir(parents=True, exist_ok=True)
    counts = pd.crosstab(d.Measure_ID, d.occupancy_class).reindex(columns=range(4), fill_value=0)
    counts.columns = ["class_0", "class_1", "class_2", "class_3"]
    groups = d.groupby("Measure_ID").agg(rows=("Occupancy", "size"), room=("Aula", "first"),
        school_candidate=("school_candidate", "first"), start=("time", "min"), end=("time", "max"),
        dependency=("candidate_leakage_group", "first"))
    groups = groups.join(counts)
    write_csv(groups.reset_index(), OUT / "measure_class_counts.csv")
    zero = groups[groups.class_0 > 0]
    write_csv(zero.reset_index(), OUT / "zero_class_sources.csv")
    ids = sorted(groups.index)
    links, evidence = [], []
    # Preserve inherited raw-data components, including bridges via excluded rows.
    for _, members in groups.groupby("dependency"):
        links.extend(combinations(members.index, 2))
    duplicates = []
    key = ["Datetime", "CO2D", "TairD", "RHD"]
    for _, block in d[d.duplicated(key, keep=False)].groupby(key):
        pairs = list(combinations(sorted(block.Measure_ID.unique()), 2))
        links.extend(pairs)
        for _, row in block.iterrows():
            duplicates.append({"source_row": row.source_row, "Measure_ID": row.Measure_ID,
                               "Datetime": row.Datetime, "Occupancy": row.Occupancy,
                               "occupancy_class": row.occupancy_class,
                               "conflicting_occupancy": block.Occupancy.nunique() > 1,
                               "conflicting_class": block.occupancy_class.nunique() > 1})
    write_csv(pd.DataFrame(duplicates), OUT / "duplicate_core_time_rows.csv")
    for a, b in combinations(ids, 2):
        x, y = groups.loc[a], groups.loc[b]
        overlap = max(x.start, y.start) <= min(x.end, y.end)
        same_school = x.school_candidate == y.school_candidate
        same_room_name = x.room == y.room
        same_dep = x.dependency == y.dependency
        if overlap or same_dep or (same_school and same_room_name):
            ax, bx = d[d.Measure_ID == a], d[d.Measure_ID == b]
            common = ax.merge(bx, on=key, suffixes=("_a", "_b"))
            evidence.append(dict(left=a, right=b, time_ranges_overlap=overlap,
                                 shared_timestamps=len(set(ax.Datetime) & set(bx.Datetime)),
                                 same_school_candidate=same_school, same_room_name=same_room_name,
                                 inherited_dependency=same_dep, equal_core_time_row_pairs=len(common),
                                 conflicting_class_pairs=int((common.occupancy_class_a != common.occupancy_class_b).sum())))
    write_csv(pd.DataFrame(evidence), OUT / "pairwise_independence_checks.csv")

    # School-day batches kept together for temporal extrapolation; school-wide
    # components used for the more conservative cross-school proposal.
    batch_links = list(links)
    for _, block in d.groupby(["school_candidate", d.time.dt.date]):
        batch_links.extend(combinations(sorted(block.Measure_ID.unique()), 2))
    batch_map = components(ids, batch_links)
    school_links = list(batch_links)
    for _, block in groups.groupby("school_candidate"):
        school_links.extend(combinations(block.index, 2))
    school_map = components(ids, school_links)
    groups["batch_unit"] = groups.index.map(batch_map)
    groups["school_unit"] = groups.index.map(school_map)
    write_csv(groups.reset_index(), OUT / "independence_units.csv")
    units = groups.groupby("school_unit")[counts.columns].sum()
    write_csv(units.reset_index(), OUT / "school_unit_class_counts.csv")
    assignment, solver = optimize(units.to_numpy())
    min9_assignment, min9_solver = optimize(units.to_numpy(), minimum_zero=9)
    _, min10_solver = optimize(units.to_numpy(), minimum_zero=10)
    proposals = {}
    if assignment is not None:
        unit_to_split = dict(zip(units.index, [SPLITS[i] for i in assignment]))
        proposals["school_holdout"] = groups.school_unit.map(unit_to_split).to_dict()
    if min9_assignment is not None:
        mapping9 = dict(zip(units.index, [SPLITS[i] for i in min9_assignment]))
        proposals["school_min9_zero"] = groups.school_unit.map(mapping9).to_dict()

    # Exhaustively search all ordered whole-batch 2-cut partitions, rejecting
    # overlapping time spans. No row-level cuts or class-based group formation.
    batch = groups.groupby("batch_unit").agg(start=("start", "min"), end=("end", "max"))
    batch = batch.join(groups.groupby("batch_unit")[counts.columns].sum()).sort_values(["start", "end"])
    chronological, best = [], None
    for i in range(1, len(batch)-1):
        for j in range(i+1, len(batch)):
            blocks = [batch.iloc[:i], batch.iloc[i:j], batch.iloc[j:]]
            if not (blocks[0].end.max() < blocks[1].start.min() and blocks[1].end.max() < blocks[2].start.min()):
                continue
            matrix = np.array([b[counts.columns].sum().to_numpy() for b in blocks])
            sizes = matrix.sum(axis=1)
            all_classes = bool((matrix > 0).all())
            score = float(np.abs(sizes / len(d) - RATIOS).sum())
            chronological.append(dict(cut_1=str(blocks[0].end.max()), cut_2=str(blocks[1].end.max()),
                train_rows=int(sizes[0]), validation_rows=int(sizes[1]), test_rows=int(sizes[2]),
                train_zero=int(matrix[0,0]), validation_zero=int(matrix[1,0]), test_zero=int(matrix[2,0]),
                all_classes=all_classes, size_l1=score))
            if all_classes and (best is None or score < best[0]):
                best = (score, blocks)
    write_csv(pd.DataFrame(chronological), OUT / "chronological_cut_search.csv")
    if best:
        mapping = {u: SPLITS[s] for s, block in enumerate(best[1]) for u in block.index}
        proposals["chronological"] = groups.batch_unit.map(mapping).to_dict()
    summaries = []
    for name, mapping in proposals.items():
        proposal = groups.copy()
        proposal["proposed_split"] = proposal.index.map(mapping)
        write_csv(proposal.reset_index(), OUT / f"candidate_{name}_groups.csv")
        # Verify every inherited dependency and whole batch remains indivisible.
        assert proposal.groupby("dependency").proposed_split.nunique().max() == 1
        assert proposal.groupby("batch_unit").proposed_split.nunique().max() == 1
        if name.startswith("school_"):
            assert proposal.groupby("school_candidate").proposed_split.nunique().max() == 1
        labeled = d.assign(split=d.Measure_ID.map(mapping))
        assert labeled.groupby(key).split.nunique().max() == 1
        for split in SPLITS:
            block = proposal[proposal.proposed_split == split]
            row = dict(proposal=name, split=split, rows=int(block.rows.sum()), fraction=float(block.rows.sum()/len(d)),
                       measure_groups=len(block), schools=int(block.school_candidate.nunique()),
                       start=str(block.start.min()), end=str(block.end.max()))
            row.update({c: int(block[c].sum()) for c in counts.columns})
            zeros = labeled[(labeled.split == split) & (labeled.occupancy_class == 0)]
            row["zero_unique_time_core"] = len(zeros.drop_duplicates(key))
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    write_csv(summary, OUT / "candidate_summary.csv")

    # Read-only feature coverage: adding sensors cannot recover excluded rows.
    y = pd.to_numeric(raw.Occupancy, errors="coerce")
    valid = np.isfinite(y) & y.between(0,60) & y.mod(1).eq(0)
    time = pd.to_datetime(raw.Datetime, errors="coerce")
    valid &= time.notna() & time.le(pd.Timestamp("2026-10-26 23:59:59"))
    feature_rows = []
    for name, cols in {"D_core": ["CO2D","TairD","RHD"], "D_core_plus_TairE": ["CO2D","TairD","RHD","TairE"],
                       "CO2D_only": ["CO2D"], "CO2C1_only_unrepaired": ["CO2C1"],
                       "CO2C2_only_unrepaired": ["CO2C2"]}.items():
        mask = valid.copy()
        for c in cols:
            _, lo, hi, _ = rule(c)
            mask &= np.isfinite(raw[c]) & raw[c].between(lo, hi)
        z = raw.loc[mask & y.eq(0)]
        feature_rows.append(dict(feature_set=name, rows=int(mask.sum()), zero_rows=len(z),
                                 zero_measure_groups=int(z.Measure_ID.nunique()), zero_schools=int(z.Measure_ID.map(school).nunique())))
    features = pd.DataFrame(feature_rows)
    write_csv(features, OUT / "feature_coverage_alternatives.csv")
    meta = {"status": "PROPOSAL_ONLY_NOT_APPROVED", "target_ratios": RATIOS.tolist(),
            "input_sha256": before, "script_sha256": sha(Path(__file__)),
            "versions": {"pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
            "school_solver": solver, "minimum_9_zero_per_split_solver": min9_solver, "minimum_10_zero_per_split_solver": min10_solver,
            "objective": "2*sum absolute row-share errors + sum absolute per-class allocation-share errors; all classes >=1 per split",
            "zero_source_school_units": units.loc[units.class_0 > 0, "class_0"].tolist(),
            "no_datasets_written": True, "no_training": True}
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    (OUT / "analysis_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# A / D_core 划分可行性分析（候选，待确认）

输入 6,116 行、39 个测量组。脚本 `python src/analyze_split.py` 只写本分析目录；未修改原始/清洗数据，未生成正式训练、验证、测试文件，未训练。输入哈希见 analysis_manifest.json。

## 先检查独立性

先沿用清洗阶段基于全部原始记录的候选依赖组（包括被排除记录形成的桥接），再复查清洗数据的同时间+D_core相同观测；这些关系必须同分区。相同时间段本身不证明同源，不同教室可以同时采集，详见 pairwise_independence_checks.csv。同校同日视为不可拆批次。

学校从 Measure_ID 第2个下划线字段提取，Gaia/Escola Gaia 按已有重复证据视作候选别名；未把不同学校的 A1/P4/4to 等同名教室合并。Aula 与 ID 中教室名称不总一致，ETSEIB 10.15 的 _1/_a 等可能是批次后缀，因此跨学校候选保守地将同校所有日期/教室整体隔离，避免依赖未经证实的物理教室映射。继承的跨校候选关联也保持同分区。仍需来源方确认这些名称与关联，不能声称已证明物理独立。

清洗数据中，同时间+D_core重复涉及 {len(duplicates)} 条记录；原始人数冲突涉及 {sum(x['conflicting_occupancy'] for x in duplicates)} 条，候选类别冲突涉及 {sum(x['conflicting_class'] for x in duplicates)} 条（均按参与行计数，不是独立事件）。详见 duplicate_core_time_rows.csv。这些记录未删除/改标签，即使同分区，也会影响权重和标签一致性。

## 无人样本来源

{table(zero.reset_index()[['Measure_ID','class_0','dependency']])}

192 条无人来自 {len(zero)} 个测量组，归并后仅 {zero.dependency.nunique()} 个依赖组；学校隔离单位的无人数量为 {sorted(units.loc[units.class_0>0,'class_0'].tolist(), reverse=True)}。其中 170/192=88.54% 集中于同一候选来源。其余只有22条，因此三个集合各至少10条无人不可行（还需分配四个不可拆来源；本次整数规划结果见 manifest）。最优的最小无人数量也至多9条：剩余来源只能按13与7+2分配。不能通过随机拆分钟记录制造独立来源。

## 候选分组结果

class_0=无人(0)，class_1=低人数(1~5)，class_2=中人数(6~15)，class_3=高人数(≥16)；fraction为集合样本占全部6,116行的比例。

{table(summary[['proposal','split','rows','fraction','measure_groups','class_0','class_1','class_2','class_3','zero_unique_time_core']])}

school_holdout：优先学校/依赖独立，整数规划在每集四类均至少1条的硬约束下，最小化比例偏差（样本量权重2，各类别分配比例权重1）。目标70/15/15不是硬约束；求解状态和最优差距见 manifest。每个组的归属、教室、学校、时间和类别数见 candidate_school_holdout_groups.csv。该方案评价跨学校外推，不是严格按时间预测未来；同一集合有多个时间段，日历范围重叠不等于同源泄漏。

特别注意：该候选验证集的2条无人来自 Gaia/Bacho kira 的同一时刻（2022-02-04 11:13），D_core和标签相同，仅Measure_ID别名不同，实际上只有1个不同的时间+传感器观测。zero_unique_time_core 仅按该键计数，不代表独立采集事件。此候选虽比例接近目标，仍不推荐直接定稿。

school_min9_zero：增加每集至少9条无人的硬约束，展示少数类覆盖与比例/训练覆盖的取舍；9是保留重复行口径下可实现的最小集合无人数量上限，仍不是足够可靠的独立事件数。本次该候选把170条无人所在来源放入测试集，训练集只剩9条（8个不同时间+传感器观测），因此也不推荐定稿，不能为了凑齐验证/测试类别而牺牲训练覆盖。该候选与前者一样保持学校和依赖组隔离；详见 candidate_school_min9_zero_groups.csv。

chronological：在同校同日批次及既有依赖完整的条件下，枚举两条时间边界，禁止集合时间区间重叠，在四类齐全方案中选择样本比例绝对偏差最小者。详见 candidate_chronological_groups.csv 和全部边界搜索 chronological_cut_search.csv。同校跨日期可以跨集合，适用于已知场景未来预测，不能用它证明跨学校/教室泛化。未建立时序窗口；后续若使用窗口或预测跨度，需要另定边界隔离带。

**结论：四类在三个集合中形式上可以齐全，但无人独立来源过少，无法同时获得接近目标比例、稳定少数类评估和强来源隔离。候选只能用于方案审阅，不建议据此宣称可靠四分类表现。** 大量无人分钟来自同一次测量，样本数不等于独立事件数。确认前优先核查重复和冲突标注，再锁定任务目标、来源映射和划分；不能依据后续测试成绩反复挑选方案。

## 是否扩展传感器

{table(features)}

当前问题首先是D_core完整观测筛选后的来源/类别覆盖，未训练所以无法判断这三个传感器的四分类判别能力。直接给D_core追加TairE等列会进一步减少完整样本，不能补回无人来源；仅放宽为CO2D也不能增加本次覆盖。替换为C1/C2单通道或设计允许不同设备缺测的输入，可能恢复更多无人组，但这些只是未经修复原值落在筛查范围内的覆盖上限，不能视为可靠清洗数据；同源重复、千倍尺度和传感器身份仍需重审。更优先补充来自多个独立教室/批次的无人观测。暂不自动扩展特征、不插补或换用B、不合并数据。

## 产物

- measure_class_counts.csv / zero_class_sources.csv：逐测量组四类及无人来源。
- independence_units.csv / school_unit_class_counts.csv：批次与学校约束及类计数。
- pairwise_independence_checks.csv / duplicate_core_time_rows.csv：独立性与冲突证据。
- candidate_*_groups.csv / candidate_summary.csv：仅组级候选归属与集合统计，无逐行正式划分。
- feature_coverage_alternatives.csv / chronological_cut_search.csv / analysis_manifest.json：覆盖、边界搜索及复现依据。
"""
    (OUT / "split_feasibility_report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print("Solver:", solver, "Minimum 10 zero:", min10_solver)


if __name__ == "__main__":
    main()
