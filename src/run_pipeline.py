"""Run the full data pipeline in dependency order; never trains a model.

Order matters: preprocess_c2.py regenerates the processed versions and candidate
plan that every later step fingerprints, so it must come first.

Exit codes are propagated as follows:
- a script that fails hard (non-zero and not a declared "expected" code) aborts the run
- check_training_readiness.py returns 2 while the label-conflict gate is closed; that is
  reported, not treated as a crash, and analysis continues
- analyze_conflicts.py returns 2 while a conflict key still needs outside information

Usage: python src/run_pipeline.py [--skip-audit]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (script, args, allowed non-zero exit codes, note)
STEPS = [
    ("audit_data.py", [], {}, "原始数据审核与图表"),
    ("preprocess.py", [], {}, "A_D_core_v1 清洗"),
    ("analyze_split.py", [], {}, "划分可行性"),
    ("analyze_zero_coverage.py", [], {}, "无人样本追踪"),
    ("preprocess_c2.py", [], {}, "C2 版本与候选分组"),
    ("check_training_readiness.py", [], {2}, "训练前审查（2=门禁阻断，预期）"),
    ("analyze_conflicts.py", [], {2}, "标签冲突归属核查（2=仍有开放问题，预期）"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-audit", action="store_true", help="跳过 audit_data.py 的重新绘图")
    args = parser.parse_args()
    steps = [s for s in STEPS if not (args.skip_audit and s[0] == "audit_data.py")]
    for script, extra, allowed, note in steps:
        path = ROOT / "src" / script
        print(f"\n=== {script} — {note}", flush=True)
        result = subprocess.run([sys.executable, str(path), *extra], cwd=ROOT)
        if result.returncode != 0 and result.returncode not in allowed:
            print(f"FAILED: {script} exit={result.returncode}", file=sys.stderr)
            return 1
        if result.returncode in allowed:
            print(f"  exit {result.returncode} 为预期状态：{note}", flush=True)
    print("\n数据管线完成；未进行模型训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
