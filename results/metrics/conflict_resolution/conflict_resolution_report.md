# C2 标签冲突核查与处置结论

## 结论一句话

184 个冲突键共 368 条记录（进入 `A_C2_CO2_v1` 的部分），**全部来自同一批复制粘贴的 C2 传感器原文**：同一时间戳、同一组三通道数值同时挂在两个不同教室名下，因此无法确认哪一条人数标签属于这批观测。本次**未修改任何标签**，只给出一个可审核的「整会话隔离」候选，并保留开放问题清单。

## 冲突键的构成

| conflict_id | key_Datetime | TairC2 | RHC2 | values_physically_plausible | sensor_attribution | retained_rows | proposed_split |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022-03-07 09:20:00 | 20576.0 | 38617.0 | False | UNATTRIBUTABLE | 2 | train |
| 2 | 2022-03-07 09:21:00 | 20586.0 | 38373.0 | False | UNATTRIBUTABLE | 2 | train |
| 3 | 2022-03-07 09:22:00 | 20616.0 | 38226.0 | False | UNATTRIBUTABLE | 2 | train |

具体到 184 个键，两类情况必须分开看。下表「记录数」均指**进入 `A_C2_CO2_v1` 的行**；原始文件里同一批键共 552 行（每个键 3 行），其中 184 行已被清洗规则排除（主要是 JM A1 的 CO₂ 越界行）。

| 情形 | 键数 | C2 内记录数 | 说明 | 处置 |
| --- | ---: | ---: | --- | --- |
| A 数值越出仪器量程 | 183 | 366 | TairC2 ≈ 2×10⁴ °C、RHC2 ≈ 3.7×10⁴ %RH，两或三方数值逐位相同，不同教室不可能同时读出同一串数字 | 观测无归属，标签无法验证 → 整会话隔离候选 |
| B 数值在物理量程内 | 1 | 2 | 如键 167（TairC2=21.1 °C、RHC2=29.12 %RH），读数本身可信，但仍同时挂在两个教室名下 | 不能排除真实观测，也不能确认归属 → 保留问题清单 |

`source_links.csv` 记录这三个 Measure_ID 之间的 C2 三通道键高度重合：P4 的全部 231 个键都在 4to 中出现，4to 与 JM A1 共有 450 个完全相同的键。真正的判据不是「CO₂ 相同但人数不同」（同浓度本来就可以对应不同人数），而是**同一时间戳、同一组三通道数值同时挂在两个不同教室名下**——两台独立传感器不会给出逐位相同的温湿度，因此这是原始文件的复制/配对错位，现有清洗没有制造这个错误，只是继承了它。

## 逐 Measure_ID 影响

| Measure_ID | proposed_split | candidate_session | dataset_rows | conflict_rows | conflict_share | conflict_keys | conflicted_class_0 | conflicted_class_1 | conflicted_class_2 | conflicted_class_3 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20220307_Escola St Julia_4to_ | train | 20220307_Escola St Julia_4to_ | 438 | 184 | 0.4201 | 184 | 0 | 151 | 0 | 33 | whole session would be quarantined under conservative quarantine |
| 20220307_Escola St Julia_P4_ | train | 20220307_Escola St Julia_4to_ | 231 | 184 | 0.7965 | 184 | 0 | 0 | 184 | 0 | whole session would be quarantined under conservative quarantine |

368 条冲突记录**全部落在候选方案的 train 分区**，验证集/测试集未被这些键直接污染。但这不等于问题已解决：若按现有标签训练，模型会从 4to 学到「同一个 3.5 万级原始值 → 1~5 人」，与 P4 的「→ 11~15 人」直接矛盾。

## 隔离候选的量级

| scenario | rows | candidate_sessions | class_0 | class_1 | class_2 | class_3 | rows_removed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current_A_C2_CO2_v1 | 8139 | 25 | 545 | 2174 | 2138 | 3282 | 0 |
| quarantine_conflicted_session | 6692 | 24 | 522 | 1435 | 1691 | 3044 | 1447 |

这是**候选方案，不是已执行的操作**：现有 `A_C2_CO2_v1.csv` 未做任何删改。若采用，需确认无人类在验证/测试集中仍有 ≥2 个候选会话支撑（当前剩余 10 个为最少）。

## 未决问题清单（open_issues.csv）

| issue | conclusion | action | resolved |
| --- | --- | --- | --- |
| same_three_channel_observation_under_3_measure_ids | sensor observation cannot be attributed to a single room | no label change; quarantine candidate only | False |
| which_occupancy_label_belongs_to_the_observation | publication metadata does not say which occupant count was measured by the C2 device | requires occupied-count record or device placement trace from the source | False |
| multi_sensor_version_scale_repair | engineering candidate only; publisher erratum not found | excluded from first-round training; unaffected by the quarantine | False |
| conflict_affects_validation_or_test | evaluation sets are not directly contaminated by these keys | record as evidence; not a substitute for resolving labels | True |

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
