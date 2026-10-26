"""Evaluate the validation-selected CO2 baseline once on the frozen test set."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay

from audit_data import table
from preprocess import sha, write_csv

ROOT = Path(__file__).resolve().parents[1]
LABELS = [0, 1, 2, 3]
NAMES = ["empty (0)", "low (1-5)", "medium (6-15)", "high (16+)"]


def metrics(y, prediction):
    return {"accuracy":float(accuracy_score(y,prediction)),
            "macro_f1":float(f1_score(y,prediction,labels=LABELS,average="macro",zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(y,prediction)),
            "per_class":classification_report(y,prediction,labels=LABELS,target_names=NAMES,output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(y,prediction,labels=LABELS).tolist()}


def save_metrics(result, name, metric_dir, figure_dir):
    (metric_dir/f"{name}_metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[dict(class_id=i,class_name=NAMES[i],**result["per_class"][NAMES[i]]) for i in LABELS]
    write_csv(pd.DataFrame(rows),metric_dir/f"{name}_per_class.csv")
    write_csv(pd.DataFrame(result["confusion_matrix"],columns=[f"pred_{i}" for i in LABELS]).rename_axis("true_class").reset_index(),metric_dir/f"{name}_confusion_matrix.csv")
    fig,ax=plt.subplots(figsize=(7.5,6),layout="constrained")
    ConfusionMatrixDisplay(confusion_matrix=pd.DataFrame(result["confusion_matrix"]).to_numpy(),display_labels=NAMES).plot(ax=ax,cmap="Blues",colorbar=False,values_format="d")
    ax.set_title(name.replace("_"," "))
    fig.savefig(figure_dir/f"{name}_confusion_matrix.png",dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="co2_baseline_v1")
    args=parser.parse_args()
    if Path(args.run_name).name!=args.run_name or args.run_name in [".",".."]:
        parser.error("run-name must be a directory name")
    directory=ROOT/"results/metrics"/args.run_name
    figure_dir=ROOT/"results/figures"/args.run_name
    selection_path=directory/"selection.json"
    selection=json.loads(selection_path.read_text(encoding="utf-8"))
    model_path=ROOT/selection["selected_model_path"]
    test_path=ROOT/selection["test_path"]
    if sha(model_path)!=selection["selected_model_sha256"] or sha(test_path)!=selection["test_sha256"]:
        raise RuntimeError("Selected model or frozen test set changed")
    output=directory/"test_evaluation.json"
    if output.exists():
        saved=json.loads(output.read_text(encoding="utf-8"))
        if saved["selection_sha256"]!=sha(selection_path):
            raise RuntimeError("Selection differs from the completed test evaluation")
        print("Test already evaluated; returning the saved result (no new prediction).")
        print(json.dumps(saved,ensure_ascii=False,indent=2)); return
    model=joblib.load(model_path)
    test=pd.read_csv(test_path)
    prediction=model.predict(test[["CO2C2"]])
    result=metrics(test.occupancy_class,prediction)
    save_metrics(result,"selected_test",directory,figure_dir)
    write_csv(pd.DataFrame({"source_row":test.source_row,"true_class":test.occupancy_class,"predicted_class":prediction}),directory/"selected_test_predictions.csv")
    payload={"selected_model":selection["selected_model"],"selection_sha256":sha(selection_path),
             "model_sha256":sha(model_path),"test_sha256":sha(test_path),"test_rows":len(test),**result}
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    comparison=pd.read_csv(directory/"validation_comparison.csv")
    splits=pd.read_csv(directory/"split_summary.csv")
    manifest=json.loads((directory/"run_manifest.json").read_text(encoding="utf-8"))
    empty=result["per_class"][NAMES[0]]
    report=f"""# CO₂ 单特征四分类基线

仅使用CO2C2。随机种子42，固定配置，不搜索参数，不重采样。原始数据及原C2/D_core版本保持不变。

## 冲突排除与划分

复核184个同时间同C2三通道却类别不一致的观测键，现有来源记录不能确定应保留哪个标签。按用户授权排除全部368条对应记录，不猜测或修改标签，得到7,771行新版本。排除日志见excluded_conflicts.csv；全部被排除行原属于候选训练集。其余行维持原候选会话归属，验证/测试没有重新平衡或重新挑选。

{table(splits)}

Measure_ID、候选会话、继承同源组、同时间CO₂键及原始C1/C2同源关联均未跨集合。核查结果和数据哈希见leakage_checks.csv、run_manifest.json。已有同源关系被隔离不等于已证明物理实验完全独立；其余无类别冲突的重复未擅自删除，仍可能影响样本权重。

## 验证集模型比较

{table(comparison.round(4))}

多数类DummyClassifier(strategy=most_frequent)；随机森林300棵树、min_samples_leaf=3、class_weight=balanced；RBF SVM C=1、gamma=scale、class_weight=balanced。三者均使用Pipeline，SVM中的StandardScaler仅在训练集拟合，其他两者只按列选择CO₂。全部配置在拟合前记录，未使用测试分数调整参数。

按验证集Macro-F1选择 **{selection['selected_model']}**（平分时依次比较Balanced Accuracy、Accuracy，再按固定模型顺序）。选定后不合并验证集重训，直接评估已保存的训练集模型，因此验证与测试对应同一管线。其他模型不做最终测试集比较。

## 选定模型的最终测试

Accuracy={result['accuracy']:.4f}，Macro-F1={result['macro_f1']:.4f}，Balanced Accuracy={result['balanced_accuracy']:.4f}。

无人类：Precision={empty['precision']:.4f}，Recall={empty['recall']:.4f}，F1={empty['f1-score']:.4f}，支持样本={int(empty['support'])}。无人类识别是否可用应以这些指标为准，不能仅看总体准确率。

{table(pd.read_csv(directory/'selected_test_per_class.csv').round(4))}

![选定模型测试混淆矩阵](../../figures/{args.run_name}/selected_test_confusion_matrix.png)

## 评估局限

训练/验证/测试有12/6/7个保守候选会话；无人样本在验证、测试中各仅来自2个会话，分钟记录并非独立实验。此划分允许同校/同教室不同日期跨集合，衡量已知场景下的新批次，不代表新学校泛化或严格未来预测。CO₂受通风、位置和滞后影响，单特征不能唯一确定人数。冲突排除改变训练分布，剩余非冲突行仍需来源核实；这些结果属于首轮基线，不代表部署性能或统计稳定性。零分母Precision/Recall按0报告，未伪造缺类指标。

## 文件与复现

模型：models/{args.run_name}/ 下的majority.joblib、random_forest.joblib、svm.joblib（完整Pipeline）。验证集每个模型、测试集选定模型的总体指标/逐类指标/混淆矩阵和预测均保存在本目录；图在results/figures/{args.run_name}/。selection.json在测试前落盘，test_evaluation.json绑定模型和选择哈希；重复评估只读取已有结果。

运行 `python src/train.py --run-name {args.run_name}`，随后 `python src/evaluate.py --run-name {args.run_name}`。已有实验不覆盖；从头复现使用新的run-name，保持同一输入、分组、配置和软件版本。实际版本：{manifest['versions']}。
"""
    (directory/"baseline_report.md").write_text(report,encoding="utf-8")
    print("Selected:",selection["selected_model"])
    print(json.dumps({k:result[k] for k in ["accuracy","macro_f1","balanced_accuracy"]},indent=2))
    print("Empty class:",empty)


if __name__=="__main__":
    main()
