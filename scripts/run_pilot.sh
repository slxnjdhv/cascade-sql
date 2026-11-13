#!/usr/bin/env bash
# 50-question pilot on BIRD calib with one local and one large model.
# Purpose (Experiment_Design.md, week 5): check the prompt/extraction works, measure real
# token counts and latency, and compare B1 accuracy with the model's published number.
#   usage: LOCAL=omnisql-7b LARGE=omnisql-32b bash scripts/run_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LOCAL=${LOCAL:-omnisql-7b}
LARGE=${LARGE:-omnisql-32b}
N=${N:-50}
MAN=data/manifests/calib.jsonl
OUT=results/pilot
export PYTHONPATH=src

python -m cascade_sql generate --model "$LOCAL" --manifest "$MAN" --limit "$N" --out "$OUT/$LOCAL/calib.seed0.jsonl"
python -m cascade_sql generate --model "$LARGE" --manifest "$MAN" --limit "$N" --out "$OUT/$LARGE/calib.seed0.jsonl"
python -m cascade_sql annotate --manifest "$MAN" --records "$OUT/$LOCAL/calib.seed0.jsonl"
python -m cascade_sql annotate --manifest "$MAN" --records "$OUT/$LARGE/calib.seed0.jsonl"

python - <<PY
import json, statistics as st
for m in ["$LOCAL", "$LARGE"]:
    rs = [json.loads(l) for l in open(f"$OUT/{m}/calib.seed0.ann.jsonl")]
    acc = sum(r["labels"].get("orig") or 0 for r in rs) / len(rs)
    print(f"{m:>20}: n={len(rs)} EX={acc:.3f}  prompt_tok/q={st.mean(r['prompt_tokens'] for r in rs):.0f}"
          f"  out_tok/q={st.mean(r['output_tokens'] for r in rs):.0f}  latency/q={st.mean(r['latency_s'] for r in rs):.1f}s"
          f"  repairs/q={st.mean(r['n_repairs'] for r in rs):.2f}"
          f"  empty_sql={sum(1 for r in rs if not r['chosen_sql'])}")
PY
