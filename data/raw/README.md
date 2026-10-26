# 原始数据（未纳入版本库）

原始文件体积较大且受第三方许可约束，不提交到仓库。请按下面说明自行放置，脚本会按固定路径读取。

## 必需文件

| 文件 | 用途 | 大小 |
| --- | --- | --- |
| `ieq-dataset.csv` | 主数据集（A），所有清洗与 C2 版本均由此生成 | 约 4 MB |
| `Classroom_data_combined_main.xlsx` | 数据集 B，备选，不参与当前流程 | 约 3 MB |

## 获取方式

`ieq-dataset.csv` 来自 Zenodo 的教室室内环境质量数据集：

- 记录页：<https://zenodo.org/records/14621677>
- 配套论文：<https://doi.org/10.1155/ina/5517242>

下载后放到本目录，文件名保持为 `ieq-dataset.csv`。

## 完整性校验

`src/preprocess_c2.py` 会用 MD5 校验主数据集：

```
4e79072861dc6c0ec9d5035769fbfb8c
```

校验不匹配时脚本不会报错退出，但会**禁用全部温湿度修复规则**（修复依赖"文件确实是发布版本"这一前提），并在 `results/metrics/c2_preprocessing/manifest.json` 中把 `published_md5_match` 记为 `false`。若你拿到的是重新导出的同名文件，请先核对这一点。

原始文件在整个流程中只读，任何脚本都不会写回本目录。
