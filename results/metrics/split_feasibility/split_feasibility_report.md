# A / D_core 划分可行性分析（候选，待确认）

输入 6,116 行、39 个测量组。脚本 `python src/analyze_split.py` 只写本分析目录；未修改原始/清洗数据，未生成正式训练、验证、测试文件，未训练。输入哈希见 analysis_manifest.json。

## 先检查独立性

先沿用清洗阶段基于全部原始记录的候选依赖组（包括被排除记录形成的桥接），再复查清洗数据的同时间+D_core相同观测；这些关系必须同分区。相同时间段本身不证明同源，不同教室可以同时采集，详见 pairwise_independence_checks.csv。同校同日视为不可拆批次。

学校从 Measure_ID 第2个下划线字段提取，Gaia/Escola Gaia 按已有重复证据视作候选别名；未把不同学校的 A1/P4/4to 等同名教室合并。Aula 与 ID 中教室名称不总一致，ETSEIB 10.15 的 _1/_a 等可能是批次后缀，因此跨学校候选保守地将同校所有日期/教室整体隔离，避免依赖未经证实的物理教室映射。继承的跨校候选关联也保持同分区。仍需来源方确认这些名称与关联，不能声称已证明物理独立。

清洗数据中，同时间+D_core重复涉及 730 条记录；原始人数冲突涉及 24 条，候选类别冲突涉及 24 条（均按参与行计数，不是独立事件）。详见 duplicate_core_time_rows.csv。这些记录未删除/改标签，即使同分区，也会影响权重和标签一致性。

## 无人样本来源

| Measure_ID | class_0 | dependency |
| --- | --- | --- |
| 20220204_Escola Gaia_Bacho kira_20220204 | 1 | candidate_3116bfa963336b0e |
| 20220204_Gaia_Bacho kira_20220204 | 1 | candidate_3116bfa963336b0e |
| 20220321_Escola Roureda_A2_ | 13 | candidate_1f24a2383aeacd5e |
| 20220322_Escola Bressol Vapor Buxeda Nou_A1_ | 97 | candidate_5e7f3926d05d648d |
| 20220322_Escola Bressol Vapor Buxeda Nou_A2_ | 73 | candidate_5e7f3926d05d648d |
| 20220525_ETSEIB_10.15_1_20220525 | 1 | candidate_4fbb86f761cd73d4 |
| 20220525_ETSEIB_10.15_a_20220525 | 6 | candidate_4fbb86f761cd73d4 |

192 条无人来自 7 个测量组，归并后仅 4 个依赖组；学校隔离单位的无人数量为 [170, 13, 7, 2]。其中 170/192=88.54% 集中于同一候选来源。其余只有22条，因此三个集合各至少10条无人不可行（还需分配四个不可拆来源；本次整数规划结果见 manifest）。最优的最小无人数量也至多9条：剩余来源只能按13与7+2分配。不能通过随机拆分钟记录制造独立来源。

## 候选分组结果

class_0=无人(0)，class_1=低人数(1~5)，class_2=中人数(6~15)，class_3=高人数(≥16)；fraction为集合样本占全部6,116行的比例。

| proposal | split | rows | fraction | measure_groups | class_0 | class_1 | class_2 | class_3 | zero_unique_time_core |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| school_holdout | train | 4173 | 0.6823086984957488 | 28 | 177 | 1120 | 1233 | 1643 | 177 |
| school_holdout | validation | 1064 | 0.17396991497710923 | 9 | 2 | 211 | 200 | 651 | 1 |
| school_holdout | test | 879 | 0.14372138652714192 | 2 | 13 | 310 | 99 | 457 | 13 |
| school_min9_zero | train | 4237 | 0.6927730542838456 | 33 | 9 | 1096 | 1124 | 2008 | 8 |
| school_min9_zero | validation | 879 | 0.14372138652714192 | 2 | 13 | 310 | 99 | 457 | 13 |
| school_min9_zero | test | 1000 | 0.16350555918901244 | 4 | 170 | 235 | 309 | 286 | 170 |
| chronological | train | 3941 | 0.644375408763898 | 27 | 15 | 1274 | 707 | 1945 | 14 |
| chronological | validation | 925 | 0.1512426422498365 | 2 | 170 | 208 | 306 | 241 | 170 |
| chronological | test | 1250 | 0.20438194898626552 | 10 | 7 | 159 | 519 | 565 | 7 |

school_holdout：优先学校/依赖独立，整数规划在每集四类均至少1条的硬约束下，最小化比例偏差（样本量权重2，各类别分配比例权重1）。目标70/15/15不是硬约束；求解状态和最优差距见 manifest。每个组的归属、教室、学校、时间和类别数见 candidate_school_holdout_groups.csv。该方案评价跨学校外推，不是严格按时间预测未来；同一集合有多个时间段，日历范围重叠不等于同源泄漏。

特别注意：该候选验证集的2条无人来自 Gaia/Bacho kira 的同一时刻（2022-02-04 11:13），D_core和标签相同，仅Measure_ID别名不同，实际上只有1个不同的时间+传感器观测。zero_unique_time_core 仅按该键计数，不代表独立采集事件。此候选虽比例接近目标，仍不推荐直接定稿。

school_min9_zero：增加每集至少9条无人的硬约束，展示少数类覆盖与比例/训练覆盖的取舍；9是保留重复行口径下可实现的最小集合无人数量上限，仍不是足够可靠的独立事件数。本次该候选把170条无人所在来源放入测试集，训练集只剩9条（8个不同时间+传感器观测），因此也不推荐定稿，不能为了凑齐验证/测试类别而牺牲训练覆盖。该候选与前者一样保持学校和依赖组隔离；详见 candidate_school_min9_zero_groups.csv。

chronological：在同校同日批次及既有依赖完整的条件下，枚举两条时间边界，禁止集合时间区间重叠，在四类齐全方案中选择样本比例绝对偏差最小者。详见 candidate_chronological_groups.csv 和全部边界搜索 chronological_cut_search.csv。同校跨日期可以跨集合，适用于已知场景未来预测，不能用它证明跨学校/教室泛化。未建立时序窗口；后续若使用窗口或预测跨度，需要另定边界隔离带。

**结论：四类在三个集合中形式上可以齐全，但无人独立来源过少，无法同时获得接近目标比例、稳定少数类评估和强来源隔离。候选只能用于方案审阅，不建议据此宣称可靠四分类表现。** 大量无人分钟来自同一次测量，样本数不等于独立事件数。确认前优先核查重复和冲突标注，再锁定任务目标、来源映射和划分；不能依据后续测试成绩反复挑选方案。

## 是否扩展传感器

| feature_set | rows | zero_rows | zero_measure_groups | zero_schools |
| --- | --- | --- | --- | --- |
| D_core | 6116 | 192 | 7 | 4 |
| D_core_plus_TairE | 3357 | 170 | 2 | 1 |
| CO2D_only | 6116 | 192 | 7 | 4 |
| CO2C1_only_unrepaired | 6032 | 325 | 10 | 2 |
| CO2C2_only_unrepaired | 8139 | 545 | 17 | 7 |

当前问题首先是D_core完整观测筛选后的来源/类别覆盖，未训练所以无法判断这三个传感器的四分类判别能力。直接给D_core追加TairE等列会进一步减少完整样本，不能补回无人来源；仅放宽为CO2D也不能增加本次覆盖。替换为C1/C2单通道或设计允许不同设备缺测的输入，可能恢复更多无人组，但这些只是未经修复原值落在筛查范围内的覆盖上限，不能视为可靠清洗数据；同源重复、千倍尺度和传感器身份仍需重审。更优先补充来自多个独立教室/批次的无人观测。暂不自动扩展特征、不插补或换用B、不合并数据。

## 产物

- measure_class_counts.csv / zero_class_sources.csv：逐测量组四类及无人来源。
- independence_units.csv / school_unit_class_counts.csv：批次与学校约束及类计数。
- pairwise_independence_checks.csv / duplicate_core_time_rows.csv：独立性与冲突证据。
- candidate_*_groups.csv / candidate_summary.csv：仅组级候选归属与集合统计，无逐行正式划分。
- feature_coverage_alternatives.csv / chronological_cut_search.csv / analysis_manifest.json：覆盖、边界搜索及复现依据。
