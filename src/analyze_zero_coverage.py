"""Trace zero occupancy loss and compare sensor/split options, read-only.

Run: python src/analyze_zero_coverage.py
Division by 1000 is a diagnostic hypothesis only; no corrected data exported.
"""
from itertools import combinations
import json

import numpy as np
import pandas as pd

from analyze_split import ROOT, school, components, optimize
from audit_data import rule, table
from preprocess import sha, write_csv, classes

OUT = ROOT / "results/metrics/zero_coverage"


def main():
    paths = [ROOT / "data/raw/ieq-dataset.csv", ROOT / "data/processed/A_D_core_v1.csv",
             ROOT / "results/metrics/preprocess_A_row_decisions.csv",
             ROOT / "results/metrics/preprocess_A_group_mapping.csv"]
    protected = [*sorted((ROOT / "data/raw").glob("*")), *sorted((ROOT / "data/processed").glob("*"))]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected if p.is_file()}
    a, cleaned, decisions, group_map = [pd.read_csv(p) for p in paths]
    OUT.mkdir(parents=True, exist_ok=True)
    y = pd.to_numeric(a.Occupancy, errors="coerce")
    t = pd.to_datetime(a.Datetime, errors="coerce").astype("datetime64[us]")
    valid = np.isfinite(y) & y.between(0, 60) & y.mod(1).eq(0) & t.notna() & t.le(pd.Timestamp("2026-10-26 23:59:59"))
    zero = y.eq(0)
    meta = a[["Measure_ID", "Aula", "Datetime"]].copy()
    meta.insert(0, "source_row", np.arange(len(a)) + 2)
    meta["school_candidate"] = a.Measure_ID.map(school)
    meta["room_candidate"] = meta.school_candidate + " / " + a.Aula.fillna("<missing>")
    meta["date"] = t.dt.strftime("%Y-%m-%d")
    meta["measure_prefix_date"] = a.Measure_ID.str[:8]
    meta["date_prefix_mismatch"] = t.dt.strftime("%Y%m%d").ne(meta.measure_prefix_date)
    meta["dependency"] = a.Measure_ID.map(group_map.set_index("Measure_ID").candidate_leakage_group)
    assert meta.dependency.notna().all()
    ids = sorted(a.Measure_ID.unique())
    links = []
    for _, block in meta.groupby("dependency"):
        links.extend(combinations(sorted(block.Measure_ID.unique()), 2))
    sensor_pairs = []
    for sensor in [1, 2]:
        key = ["Datetime", f"CO2C{sensor}", f"TairC{sensor}", f"RHC{sensor}"]
        duplicate = a[key].notna().all(axis=1) & a.duplicated(key, keep=False)
        for _, block in a.loc[duplicate].groupby(key):
            for left, right in combinations(sorted(block.Measure_ID.unique()), 2):
                links.append((left, right))
                sensor_pairs.append(dict(sensor=f"C{sensor}", left=left, right=right,
                                         matched_timestamp_keys=1,
                                         known_label_conflict=int(block.Occupancy.dropna().nunique()>1)))
    pair_table = pd.DataFrame(sensor_pairs).groupby(["sensor", "left", "right"], as_index=False).sum()
    write_csv(pair_table, OUT / "C1_C2_cross_measure_matches.csv")
    # Whole school-day batches are conservative proxies, not verified sessions.
    for _, block in meta.groupby(["school_candidate", "date"]):
        links.extend(combinations(sorted(block.Measure_ID.unique()), 2))
    sessions = components(ids, links)
    meta["session_candidate"] = a.Measure_ID.map(sessions)
    school_links = list(links)
    for _, block in meta.groupby("school_candidate"):
        school_links.extend(combinations(sorted(block.Measure_ID.unique()), 2))
    schools = components(ids, school_links)
    meta["school_unit"] = a.Measure_ID.map(schools)
    retained = meta.source_row.isin(cleaned.source_row)
    trace = meta.loc[zero].copy()
    trace["D_core_retained"] = retained.loc[zero]
    trace["exclusion_reason"] = trace.source_row.map(decisions.set_index("source_row").exclusion_reason).fillna("retained")
    for c in ["CO2D", "TairD", "RHD", "CO2C1", "TairC1", "RHC1", "CO2C2", "TairC2", "RHC2"]:
        trace[c] = a.loc[zero, c]
        _, lo, hi, _ = rule(c)
        trace[c + "_raw_range_ok"] = np.isfinite(a.loc[zero,c]) & a.loc[zero,c].between(lo,hi)
    write_csv(trace, OUT / "zero_row_trace.csv")
    write_csv(trace.groupby(["D_core_retained", "exclusion_reason"]).size().reset_index(name="rows"), OUT / "zero_loss_reasons.csv")
    summaries = {}
    for name, keys in {"school": ["school_candidate"], "room": ["school_candidate", "Aula"],
                       "measure": ["Measure_ID", "school_candidate", "Aula"], "date": ["date"],
                       "school_date": ["school_candidate", "date"], "session": ["session_candidate"]}.items():
        g = trace.groupby(keys, dropna=False).agg(raw_zero=("source_row","size"), retained_zero=("D_core_retained","sum"),
                measure_ids=("Measure_ID","nunique"), dependency_units=("dependency","nunique"),
                candidate_sessions=("session_candidate","nunique"), days=("date","nunique"))
        g["lost_zero"] = g.raw_zero-g.retained_zero
        summaries[name] = g.reset_index()
        write_csv(summaries[name], OUT / f"zero_by_{name}.csv")

    # Unit hypotheses, per-field/per-measure evidence, simultaneous D references.
    scale_rows, scale_groups = [], []
    ok, candidate = {}, {}
    for c in ["CO2D", "TairD", "RHD", "CO2C1", "TairC1", "RHC1", "DewPointC1", "CO2C2", "TairC2", "RHC2", "DewPointC2"]:
        _, lo, hi, unit = rule(c)
        x = a[c]
        ok[c] = np.isfinite(x) & x.between(lo, hi)
        candidate[c] = np.isfinite(x) & ~ok[c] & x.abs().ge(1000) & (x/1000).between(lo, hi)
        if c in ["CO2D", "TairD", "RHD"]:
            continue
        reference = "CO2D" if c.startswith("CO2") else "TairD" if c.startswith("Tair") else "RHD" if c.startswith("RH") else None
        result = dict(field=c, assumed_unit=unit, raw_missing=int(x.isna().sum()), raw_in_range=int(ok[c].sum()),
                      divide1000_candidates=int(candidate[c].sum()), zero_raw_ok=int((zero & ok[c]).sum()),
                      zero_scale_candidates=int((zero & candidate[c]).sum()), reference=reference or "none",
                      reference_pairs=0, median_abs_difference_before=None, median_abs_difference_hypothesis=None)
        if reference:
            _, rlo, rhi, _ = rule(reference)
            paired = candidate[c] & a[reference].between(rlo,rhi) & np.isfinite(a[reference])
            result["reference_pairs"] = int(paired.sum())
            if paired.any():
                result["median_abs_difference_before"] = float((x[paired]-a.loc[paired, reference]).abs().median())
                result["median_abs_difference_hypothesis"] = float((x[paired]/1000-a.loc[paired, reference]).abs().median())
        mixed, switches = 0, 0
        for group, indices in a.groupby("Measure_ID").groups.items():
            normal = int(ok[c].loc[indices].sum()); scaled = int(candidate[c].loc[indices].sum())
            mixed += int(normal > 0 and scaled > 0)
            # Only adjacent observations <=2 minutes, never across group boundaries.
            order = t.loc[indices].sort_values().index
            state = pd.Series(np.where(ok[c], 0, np.where(candidate[c], 1, np.nan)), index=a.index).loc[order]
            dt = t.loc[order].diff().dt.total_seconds()
            changes = int((state.notna() & state.shift().notna() & state.ne(state.shift()) & dt.gt(0) & dt.le(120)).sum())
            switches += changes
            scale_groups.append(dict(field=c, Measure_ID=group, raw_normal=normal, scale_candidates=scaled,
                                     mixed_scales=normal > 0 and scaled > 0, adjacent_scale_switches=changes))
        result["mixed_scale_measure_groups"] = mixed
        result["adjacent_scale_switches"] = switches
        scale_rows.append(result)
    scale = pd.DataFrame(scale_rows)
    write_csv(scale, OUT / "scale_hypothesis_evidence.csv")
    write_csv(pd.DataFrame(scale_groups), OUT / "scale_evidence_by_measure.csv")

    masks = {}
    for name, cols in {"D_core": ["CO2D","TairD","RHD"], "C1_CO2_raw": ["CO2C1"], "C2_CO2_raw": ["CO2C2"],
                       "C1_core_raw": ["CO2C1","TairC1","RHC1"], "C2_core_raw": ["CO2C2","TairC2","RHC2"]}.items():
        masks[name] = valid & pd.concat([ok[c] for c in cols], axis=1).all(axis=1)
        if name in ["C1_core_raw", "C2_core_raw"]:
            masks[name.replace("raw", "hypothesis_upper_bound")] = valid & pd.concat([ok[c] | candidate[c] for c in cols], axis=1).all(axis=1)
    masks["any_CO2_coverage_only"] = valid & (ok["CO2D"] | ok["CO2C1"] | ok["CO2C2"])
    coverage, distributions = [], []
    for name, mask in masks.items():
        z = mask & zero
        coverage.append(dict(scenario=name, valid_rows=int(mask.sum()), zero_rows=int(z.sum()),
                             recovered_excluded_zero=int((z & ~retained).sum()), zero_measure_ids=int(a.loc[z,"Measure_ID"].nunique()),
                             date_prefix_mismatch_rows=int((mask & meta.date_prefix_mismatch).sum()),
                             zero_schools=int(meta.loc[z,"school_candidate"].nunique()),
                             zero_dependency_units=int(meta.loc[z,"dependency"].nunique()),
                             zero_candidate_sessions=int(meta.loc[z,"session_candidate"].nunique())))
        for school_name, block in meta.loc[z].groupby("school_candidate"):
            distributions.append(dict(scenario=name, school=school_name, zero_rows=len(block),
                                      candidate_sessions=block.session_candidate.nunique()))
    coverage = pd.DataFrame(coverage)
    write_csv(coverage, OUT / "sensor_coverage_scenarios.csv")
    write_csv(pd.DataFrame(distributions), OUT / "scenario_zero_school_distribution.csv")

    # Quantify structural feasibility only; assignments from solver are discarded.
    feasibility = []
    for scenario in ["D_core", "C2_CO2_raw"]:
        mask = masks[scenario]
        subset = meta.loc[mask].copy()
        subset["class_id"] = classes(y.loc[mask])
        for level in ["school_unit", "session_candidate"]:
            mat = pd.crosstab(subset[level], subset.class_id).reindex(columns=range(4), fill_value=0)
            for minimum in [1, 10, 20]:
                assignment, status = optimize(mat.to_numpy(), minimum_zero=minimum)
                feasibility.append(dict(scenario=scenario, split_unit=level, units=len(mat),
                                        zero_units=int(mat[0].gt(0).sum()), min_zero_per_subset=minimum,
                                        feasible=assignment is not None, solver_status=status["status"],
                                        limitation="coverage only; no device/label provenance confirmation; ratios soft"))
        # Chronological necessary/constructive check over whole session spans.
        sessions_df = subset.groupby("session_candidate").agg(start=("Datetime","min"), end=("Datetime","max"))
        sessions_df = sessions_df.join(pd.crosstab(subset.session_candidate, subset.class_id).reindex(columns=range(4),fill_value=0)).sort_values("start")
        best_min_zero, feasible_cuts = -1, 0
        for i in range(1,len(sessions_df)-1):
            for j in range(i+1,len(sessions_df)):
                blocks=[sessions_df.iloc[:i],sessions_df.iloc[i:j],sessions_df.iloc[j:]]
                if not (blocks[0].end.max()<blocks[1].start.min() and blocks[1].end.max()<blocks[2].start.min()):
                    continue
                matrix=np.array([b[list(range(4))].sum().to_numpy() for b in blocks])
                if (matrix>0).all():
                    feasible_cuts+=1
                    best_min_zero=max(best_min_zero,int(matrix[:,0].min()))
        feasibility.append(dict(scenario=scenario,split_unit="chronological_whole_session",units=len(sessions_df),
                                zero_units=int(subset.loc[subset.class_id.eq(0),"session_candidate"].nunique()),
                                min_zero_per_subset=best_min_zero,feasible=feasible_cuts>0,solver_status="exhaustive",
                                limitation=f"min_zero field is maximum attainable minimum; {feasible_cuts} all-class two-cut options; ratios not constrained"))
    feas = pd.DataFrame(feasibility)
    write_csv(feas, OUT / "split_structural_feasibility.csv")
    assert zero.sum()==len(trace) and int(trace.D_core_retained.sum())==int((retained & zero).sum())
    assert masks["D_core"].equals(retained)
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    manifest={"status":"ANALYSIS_ONLY", "input_hashes_unchanged":hashes,
              "script_sha256":sha(__import__('pathlib').Path(__file__)),
              "rules":"same finite integer label and fixed as-of 2026-10-26 as preprocess; audit sensor bounds; no transformed exports",
              "units":"schools/rooms parsed from metadata, sessions are conservative school-day + inherited dependency components, not proven independent",
              "versions":{"pandas":pd.__version__,"numpy":np.__version__},"trained":False,"final_split_created":False}
    (OUT/"analysis_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    report=f"""# 无人样本追踪与数据改进建议

## 结论

**D_core 丢失无人样本的直接原因是设备通道缺失，不能靠调整划分算法解决。优先调查 C2 CO₂ 通道及标注来源，形成待审核的 A 扩展版本；目前不建议固定70/15/15并宣称可靠四分类。** 未训练，无法判断传感器本身的分类能力。仅凭四类齐全或每类20条，也不能证明评估可靠。

## 556 → 192 的逐条追踪

原始无人 {int(zero.sum())} 条，保留 {int((retained&zero).sum())} 条，排除 {int((~retained&zero).sum())} 条；排除的364条全部为 CO2D/TairD/RHD 同时缺失。无人标签无非法/未来时间排除，也未因重复删除。具体来源：ETSEIB 325条、Escola Bressol Joan Montllor 29条、Institut Xalest 9条、Escola Catalunya 1条。ETSEIB损失集中在2021-11-19至2021-11-26的6个Measure_ID。完整 source_row、原因及各通道原值见 zero_row_trace.csv。

{table(summaries['school'])}

学校名称只对已有别名 Gaia/Escola Gaia 归一；教室使用学校+Aula组合，不把跨校同名A1/4to合并。Aula与Measure_ID中的教室代号有冲突或批次后缀，真实物理教室数不能据此确认。zero_by_room/measure/date/school_date/session.csv 分别列出各级数量。

“独立测量”目前只能保守近似：先保留原审核中跨Measure_ID的重复关联（含被排除行形成的桥），再把同校同日整批绑定为候选会话。不同日期仍可能共享教室、设备和课表，不能宣称完全独立。候选会话数比原始ID数更适合衡量来源覆盖，但仍需来源方确认。D_core的192条无人只有4个候选来源；170条集中同一来源，Gaia的2条实际为同一观测的别名重复。

此外重新检查原始C1/C2的“相同时间＋同通道CO₂/温度/湿度三值完全一致”并纳入候选依赖，见 C1_C2_cross_measure_matches.csv。C2中 St Julia 与 Joan Montllor 的部分ID间出现450个相同观测键，进一步支持既有跨ID依赖，不能按不同学校字符串直接分开；不把单个CO₂数值相同或日历时间重叠当作同源证明。对存在日期前缀冲突的记录继续标记，覆盖统计不是已批准清洗结果。

## 传感器扩展：覆盖与可信度分开

{table(coverage)}

raw表示原值通过原审核范围（不是已证明正确），hypothesis_upper_bound表示每个字段原值通过或满足“越界、绝对值≥1000、除1000后在范围内”的理论上限，**不是可用清洗数据**。any_CO2仅为任一通道覆盖并集，未将传感器拼接为统一特征，不能忽略设备身份/位置/时间响应差异。

C2 CO₂单通道可覆盖545条无人，缺少的11条均为CO2C2缺失，无其他零人数CO₂越界损失。相对D_core多覆盖353条无人，来自7所学校、17个Measure_ID、11个保守候选会话；原始556条无人也分布于这17个ID、11个日期。C2覆盖的327条来自ETSEIB、170条来自Vapor Buxeda Nou，两校合计497/545=91.19%，来源集中仍然严重。完整C2三通道原值仅覆盖7条无人，主要受温湿度量纲问题限制。C1扩展大多集中ETSEIB，增加分钟记录不一定增加跨学校来源。C2覆盖多个学校/批次，更值得优先核查；必须重新审查同源记录和冲突标签，不能直接把545当独立无人事件。

## 千倍量纲修复依据及不足

{table(scale[['field','raw_in_range','divide1000_candidates','zero_scale_candidates','mixed_scale_measure_groups','adjacent_scale_switches','reference_pairs','median_abs_difference_hypothesis']])}

存在同一Measure_ID内正常量级与千倍候选交替，且部分相邻≤2分钟观测切换量级，支持小数解析/尺度混用假设，反对整列统一缩放。对候选行，同时间D设备原值合理时比较修复假设与参考的绝对差，明细同时保存假设前后中位差。温度/湿度接近D通道只能提供辅助证据，设备位置和滞后不同，CO₂也不能强求相等；无参考的候选不能据此验证。这里仅做临时算术诊断，未导出修复后的逐行值或应用缩放。

下一步需取回源Excel/导出流程、分隔符/小数格式、设备型号与单位、设备通道对应关系和会话记录。逐字段逐批次核对原始字符串及至少部分有参考观测，确认规则适用范围；无法确认的行继续隔离。CO₂已有合理范围值也不自动证明尺度正确。规则和阈值应由来源证据决定，不能根据后续测试表现选择；拟合的校准/插补仅用训练分区。

## 划分方式与可支持的泛化目标

| 方式 | 绑定条件 | 可评估目标 | 不能据此声称 |
| --- | --- | --- | --- |
| 学校留出 | 同校全部日期/教室＋跨校同源候选整体隔离 | 新学校/环境外推 | 当前少数来源足以代表所有学校；学校映射已证实 |
| 候选会话留出 | 重复依赖＋同校同日整批不拆；物理教室可复用 | 已知学校/设备分布下的新测量批次 | 新学校或新教室泛化；会话之间完全独立 |
| 时间外推 | 整会话按真实时间排序，边界不穿过依赖组 | 未来日期的实际部署表现 | 跨学校泛化；可以忽略季节/设备覆盖漂移 |

结构可行性结果如下；学校/会话行中的阈值是每集最少无人行，另外三类至少1条，70/15/15为软目标。时间行的阈值是穷举可行时间边界后，三集最小无人行数可达到的最大值，不是固定比例方案。所有求解产生的临时归属均丢弃，仅输出可行性，**未生成新候选/正式划分清单**。

{table(feas[['scenario','split_unit','units','zero_units','min_zero_per_subset','feasible']])}

D_core的学校/会话来源不足问题依然存在；已有学校方案验证集仅2条同源无人，无法稳定估计召回。C2单通道扩展若通过来源和标签核查，可增加早期ETSEIB批次及其他学校无人覆盖；“结构上可行”仍不代表质量已过关或统计精度足够。跨学校优先用学校留出/按学校的外层交叉验证；会话策略用于已知场景新批次目标，并明确共享教室。时间策略需要检查早期ETSEIB与后期其他学校的混杂；引入时间窗后按窗口和预测跨度设置边界隔离，不能随机拆同会话分钟记录。

## 推荐执行顺序与替代方案

1. 优先核查原始标签、24条参与同时间同D_core却类别冲突的记录和测量别名；不以传感器预测值重写人数标签。确认学校/物理教室/设备/会话映射及重复处理原则。
2. 以C2 CO₂单通道的更广覆盖作为下一版数据设计的优先调查对象，记录与D_core重叠及新增来源。需要温湿度时先完成C2量纲溯源；不盲目追加要求完整的列，也不把不同通道无条件当同一传感器。此次未创建扩展清洗版。
3. 补采多个独立学校、教室和日期的无人观测，兼顾占用转空置后的CO₂滞后；以独立批次数和每类来源覆盖确定规模，避免只延长同一次无人记录。来源确认后再预注册任务、主评估方式及误差/不确定性报告。
4. 当前D_core可作探索性人数回归或序数任务，并按人数区间、学校/会话报告误差；它们不消除无人覆盖缺陷。若业务允许，可将0~5合为低人数，做0~5/6~15/≥16三分类，从而不单独评估无人识别，必须明确改变任务。有人/无人二分类仍受相同少数来源限制，不能当作已解决。没有独立验证来源的四分类折应明确标记不可评估，不将缺类指标补0或靠随机切分补齐。
5. 数据扩展完成前，保留四分类为探索性目标，暂停正式三集定稿；交叉验证也不能创造新来源，外层按来源留出、内层仅用训练来源调参，不能通过反复查看测试表现筛选划分。

复现：`python src/analyze_zero_coverage.py`。输出均位于 results/metrics/zero_coverage/，输入哈希和脚本版本见 analysis_manifest.json。原始A/B及现有清洗文件均未修改，无模型训练、插补、缩放应用或最终划分。
"""
    (OUT/"zero_coverage_report.md").write_text(report,encoding="utf-8")
    print(coverage.to_string(index=False))
    print(feas[['scenario','split_unit','min_zero_per_subset','feasible']].to_string(index=False))


if __name__ == "__main__":
    main()
