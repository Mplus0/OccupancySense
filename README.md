# OccupancySense

用教室环境传感器数据估计人数的大数据课程项目。

**当前状态：数据处理阶段。**已完成数据审核、清洗、候选数据划分和训练前审查；模型训练被前置审查阻断，尚未产出正式模型结果。

## 数据来源

| 数据集 | 来源 | 说明 |
| --- | --- | --- |
| A（主） | [Zenodo record 14621677](https://zenodo.org/records/14621677) | 教室室内环境质量与人数数据集，配套论文 [10.1155/ina/5517242](https://doi.org/10.1155/ina/5517242) |
| B（备选） | `Classroom_data_combined_main.xlsx` | 暂不与 A 合并，不参与当前流程 |

原始文件不入库，获取方式与 MD5 校验见 [`data/raw/README.md`](data/raw/README.md)。

每个 `Measure_ID` 是一次教室观测会话，包含同一时间的多路传感器读数与人数标签 `Occupancy`。

## 数据版本

| 文件（`data/processed/`，可由脚本重建） | 默认特征 | 样本数 | 无人样本 |
| --- | --- | ---: | ---: |
| `A_C2_CO2_v1.csv` | CO2C2 | 8,139 | 545 |
| `A_D_core_v1.csv` | CO2D、TairD、RHD | 6,116 | 192 |
| `A_C2_core_raw_v1.csv` | CO2C2、TairC2、RHC2 原值 | 69 | 7 |
| `A_C2_core_supported_candidate_v1.csv` | CO2C2、经证据筛查的 TairC2/RHC2 | 569 | 25 |

**当前主数据集为 `A_C2_CO2_v1`。** 候选四分类：无人（0）、低人数（1～5）、中人数（6～15）、高人数（≥16）。原始人数保留在 `Occupancy` 列。

`data/processed/` 下另有一个 `A_C2_CO2_no_conflicts_v1.csv`（7,771 行 = 8,139 − 368），是早期手工剔除冲突行后的试验版本，**不由任何清洗脚本生成**，仅被 `train.py` 引用。它不是经过审核的数据集，冲突行的剔除属于上述未决问题的一部分，不应作为正式方案。

## 目录结构

```
OccupancySense/
├── src/                        脚本，按编号顺序执行
│   ├── audit_data.py           原始数据审核与图表
│   ├── preprocess.py           A_D_core_v1 清洗
│   ├── analyze_split.py        划分可行性分析
│   ├── analyze_zero_coverage.py 无人样本来源追踪
│   ├── preprocess_c2.py        C2 数据版本与候选分组（主数据集由此生成）
│   ├── check_training_readiness.py  训练前审查门禁
│   ├── analyze_conflicts.py    标签冲突归属核查
│   ├── run_pipeline.py         按依赖顺序跑通上述全部步骤
│   ├── train.py / evaluate.py  模型训练与评估（当前门禁阻断，未使用）
│   └── test_*.py               与各脚本对应的回归测试
├── data/
│   ├── raw/                    原始数据（未入库，见其中 README）
│   └── processed/              清洗产物（可由脚本重建，未入库）
├── results/
│   ├── data_audit_report.md    数据审核报告
│   ├── metrics/                统计表、检查结果、manifest 与各阶段报告
│   └── figures/                审核图表
├── models/                     模型二进制（未入库）
├── app/  web3d/                展示模块（尚未开发）
├── requirements.txt
└── README.md
```

## 数据处理流程

```
原始 ieq-dataset.csv（只读）
        │
        ├─► audit_data.py ──────► 质量审核、缺失/重复/异常诊断，图表与报告
        │
        ├─► preprocess.py ──────► A_D_core_v1（D 设备三特征）
        │
        ├─► preprocess_c2.py ───► A_C2_CO2_v1 等 C2 版本 + 候选分组
        │        （规则：缺失/非法/越界行排除；温湿度越界仅在
        │          过去锚点 + 同时刻 D 参考 + 唯一因子三者同时满足时
        │          才修复，且只写入多传感器候选版本；CO2C2 与标签不改）
        │
        ├─► analyze_split.py ───► 候选会话/学校单元划分可行性
        ├─► analyze_zero_coverage.py ► 无人样本在各场景下的覆盖
        │
        ├─► check_training_readiness.py ► 训练门禁（哈希、泄漏、标签冲突）
        └─► analyze_conflicts.py ──────► 冲突键归属结论与开放问题清单
```

几条贯穿全程的约束：

- **原始数据只读。**每个脚本运行前后都会比对原始文件与已有产物的 SHA-256，变更即报错。
- **不做标签修补。**可疑重复保留并打标记，不按多数票删行或重标。
- **划分按整组进行。**同一 `Measure_ID`、同一候选会话、同源依赖组不会跨集合。
- **未做插补与标准化。**这些参数只能在正式训练分区内拟合，属于后续步骤。

## 运行

在项目根目录执行，Python 3.11+。

```powershell
pip install -r requirements.txt

# 一键跑通全部数据处理步骤（不训练模型）
python src/run_pipeline.py

# 或单独执行
python src/audit_data.py --as-of 2026-10-26
python src/preprocess.py --as-of 2026-10-26
python src/analyze_split.py
python src/analyze_zero_coverage.py
python src/preprocess_c2.py
python src/check_training_readiness.py
python src/analyze_conflicts.py

# 回归测试
python -m unittest discover -s src -p "test_*.py"
```

**关于退出码**：`check_training_readiness.py` 与 `analyze_conflicts.py` 在存在未解决标签冲突时返回 **2**，这是设计行为，不是脚本故障。`run_pipeline.py` 已把 2 视为预期状态并继续执行。

**关于可复现性**：脚本路径均以项目目录定位，可从任意工作目录调用；审核日期固定在 `2026-10-26`；C2 修复规则、输入文件哈希与求解器状态写入 `results/metrics/*/manifest.json`。重复执行 `preprocess_c2.py` 得到的 `A_C2_CO2_v1.csv` 哈希稳定不变。

## 当前阻断问题：C2 标签冲突

`A_C2_CO2_v1` 中有 **368 条记录、184 个冲突键**：同一时间戳、同一组 CO2C2/TairC2/RHC2 数值同时挂在不同教室的 `Measure_ID` 名下，而对应的人数类别不同。

核查结论（详见[冲突核查报告](results/metrics/conflict_resolution/conflict_resolution_report.md)）：

- 冲突全部集中在 **St Julia 4to 与 St Julia P4** 两个测点，Bressol JM A1 部分行已被清洗排除；
- 两个测点的 `CO2C2` 与 `Datetime` 逐点相同，其中 183 个键的温湿度是 `2×10⁴ °C` / `3.7×10⁴ %RH` 这类越出仪器量程的数值——不同教室的独立传感器不可能给出逐位相同的读数，这是**原始文件的复制/配对错位**；
- 但**无法确认错位发生在哪一侧**，因此没有修改任何标签；
- 备选处置「整会话隔离」量化后为：8,139 → 6,692 行，四类样本均仍有足够会话支撑。这是候选方案，**尚未执行**；
- 冲突行全部位于候选方案的 train 分区，验证集/测试集未被直接污染。

需要外部信息才能推进：该时段课堂人数记录、C2 设备与教室的分配表、发布方更正说明。未决项见 [`open_issues.csv`](results/metrics/conflict_resolution/open_issues.csv)。

## 主要报告

- [数据审核](results/data_audit_report.md)
- [C2 数据质量与候选划分](results/metrics/c2_preprocessing/quality_report.md)
- [训练前审查](results/metrics/pretraining_audit/training_readiness_report.md)
- [标签冲突核查](results/metrics/conflict_resolution/conflict_resolution_report.md)
- [无人样本与特征覆盖](results/metrics/zero_coverage/zero_coverage_report.md)
- [划分可行性](results/metrics/split_feasibility/split_feasibility_report.md)

## 后续

1. 取得冲突时段的课堂人数或设备分配记录，确定标签归属，使训练门禁通过；
2. 采纳正式划分方案（会话级或学校级），并在分区内拟合插补/标准化；
3. 训练基线模型（多数类 / 随机森林 / SVM）并做对比；
4. 开发 `app/` 与 `web3d/` 展示模块。

时间、教室、`Measure_ID`、候选会话、原值和审核标记仅用于追踪与分组，不作为预测特征；类别与人数是目标列，不能作为输入。
