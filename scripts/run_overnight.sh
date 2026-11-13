#!/usr/bin/env bash
# Unattended laptop run (Revised_Plan_Base_Paper.md). Resumable: re-running skips finished questions.
# Keep the Mac plugged in with the lid open. Progress: tail -f results/overnight.log
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
LOG=results/overnight.log
mkdir -p results
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

gen() {  # model split
  say "generate $1 $2"
  $PY -m cascade_sql generate --model "$1" --manifest "data/manifests/$2.jsonl" >> "$LOG" 2>&1 || say "  !! generate $1 $2 failed"
  say "annotate $1 $2"
  $PY -m cascade_sql annotate --manifest "data/manifests/$2.jsonl" --records "results/$1/$2.seed0.jsonl" >> "$LOG" 2>&1 || say "  !! annotate $1 $2 failed"
}

say "===== START overnight run ====="
SMALL=qwen-coder-1.5b-mlx
for sp in test_b calib g_train; do gen $SMALL $sp; done

# --- choose the escalation target: 7B if it is fast enough on this Mac, else 3B
say "speed check omnisql-7b-mlx"
$PY scripts/bench_mlx.py --model sazak/OmniSQL-7B-4bit-mlx --n 5 --styles maleki > results/bench_7b_maleki.log 2>&1
SPQ=$($PY -c "import json;print([json.loads(l) for l in open('results/bench_7b_maleki.log') if l.startswith('{')][-1]['sec_per_q'])" 2>/dev/null || echo 9999)
say "  7B sec/question = $SPQ"
LARGE=omnisql-7b-mlx
if [ "$(echo "$SPQ > 90" | bc)" = "1" ]; then LARGE=qwen-coder-3b-mlx; say "  7B too slow -> using $LARGE"; fi
for sp in test_b calib; do gen $LARGE $sp; done

# --- analysis for the main pair
say "reproduce (R0) + fit gate + evaluate"
for sp in calib test_b; do $PY -m cascade_sql reproduce --model $SMALL --split $sp >> "$LOG" 2>&1; done
for lab in arcwise bird_official; do $PY -m cascade_sql reproduce --model $SMALL --split test_b --label $lab >> "$LOG" 2>&1; done
$PY -m cascade_sql fit-gate --local $SMALL >> "$LOG" 2>&1 || say "  !! fit-gate failed"
$PY -m cascade_sql evaluate --local $SMALL --large $LARGE --splits test_b >> "$LOG" 2>&1 || say "  !! evaluate failed"
echo "$LARGE" > results/large_model_used.txt

# --- ablation: tiny model (lowest priority, runs last)
TINY=qwen-coder-0.5b-mlx
for sp in test_b calib g_train; do gen $TINY $sp; done
$PY -m cascade_sql reproduce --model $TINY --split test_b >> "$LOG" 2>&1
$PY -m cascade_sql fit-gate --local $TINY >> "$LOG" 2>&1 && \
$PY -m cascade_sql evaluate --local $TINY --large $LARGE --splits test_b >> "$LOG" 2>&1
say "===== DONE ====="
