# A / D_core 第一版清洗记录

运行：`python src/preprocess.py --as-of 2026-10-26`。固定规则与输入生成确定性结果，保留原始行顺序；source_row 为源文件记录号（含表头，首条数据=2）。原始文件 SHA-256 运行前后一致。B 未读取、未合并。

输入 11,085 行，输出 **6,116 行**，排除 4,969 行；保留 39 个测量组。输出 `data/processed/A_D_core_v1.csv`。

## 处理数量

| rule | reason | matched_raw_rows | input_rows | removed_this_step | remaining_rows |
| --- | --- | --- | --- | --- | --- |
| missing_label | 人数标签缺失；不当作0或插补 | 1237 | 11085 | 1237 | 9848 |
| invalid_label | 人数必须为有限整数且在0~60内 | 0 | 9848 | 0 | 9848 |
| invalid_time | 时间缺失或不符合YYYY-MM-DD HH:MM:SS | 0 | 9848 | 0 | 9848 |
| future_time | 晚于固定审核日期；隔离，不推测修正年份 | 2 | 9848 | 0 | 9848 |
| missing_D_core | D_core不完整；v1采用完整观测，不插补 | 4711 | 9848 | 3732 | 6116 |
| invalid_D_core | D_core包含非数字或非有限值 | 0 | 6116 | 0 | 6116 |
| out_of_range_D_core | 超出预先确定的审核范围；隔离待核实，不裁剪 | 0 | 6116 | 0 | 6116 |

matched_raw_rows 可重叠；removed_this_step 按表中顺序互斥，之和等于总排除数。每行所有匹配规则及第一排除原因见 `preprocess_A_row_decisions.csv`，排除行的原始全部字段见 `preprocess_A_excluded_rows.csv`。

## 候选类别

| class_id | class_name | before_valid_labels | after | after_fraction |
| --- | --- | --- | --- | --- |
| 0 | 无人 | 556 | 192 | 0.031393067364290386 |
| 1 | 低人数 | 2514 | 1641 | 0.2683126226291694 |
| 2 | 中人数 | 2662 | 1532 | 0.25049051667756705 |
| 3 | 高人数 | 4116 | 2751 | 0.4498037933289732 |

分类阈值由任务指定，不从数据学习。before_valid_labels 仅统计原始合法标签。组×类别数量含清洗后为0的原始组，见 `preprocess_A_group_class_counts.csv`。

## 使用约定与遗留问题

- 默认输入严格限定为 `CO2D, TairD, RHD`；范围沿用审核报告（250~10000 ppm、−50~60 °C、0~100%），属于筛查假设，边界包含端点。未依据全量分位数删除离群值，未插补、标准化或修复量纲。
- 人数、候选分类及其中文名称均为目标相关字段，不作为输入。source_row、Datetime、Aula、Measure_ID、候选依赖组和审核标记仅为追踪/分组元数据，不作为预测特征。未对缺失教室或组标识猜测填充。
- 保留输出中的 2687 条待核查记录；它们符合标签、时间和D_core基本条件，但不代表已确认独立。完整人工清单见 `preprocess_A_manual_review.csv`（包括被其他规则排除的记录，并标记 retained）。同名教室+时间碰撞只是候选；不同学校可能同名。
- `preprocess_A_measure_links.csv` 和 `preprocess_A_group_mapping.csv` 记录疑似同源依赖。candidate_leakage_group 为保守连通组，只供后续避免已知重叠跨分区，不证明实际同源，也不能替代人工确认学校/物理教室。未合并原始 Measure_ID，未删除任何疑似重复，未执行最终划分。即使相关组同分区，重复样本仍可能影响训练权重及评估，必须核查后确定处理。
- 日期与ID前缀不一致但时间本身合法的记录保留审核，不擅自改日期；明显未来时间隔离。非D_core传感器不进入此版训练输入，其原值仍在原始文件，可通过 source_row 追溯。千倍候选逐行清单与规则见 `preprocess_A_scale_candidate_*.csv`，全部标记未应用。
- 完整观测筛选改变了设备、教室和类别覆盖，不能把本版当作原始A的无偏代表；见前后对比及组统计。后续先核查重叠组和时间冲突，再确定分组/时间划分；需要拟合的处理仅用训练分区。此次未训练模型。

可复现清单、特征白名单、输入/输出/脚本哈希和实际依赖版本见 `preprocess_A_manifest.json`。
