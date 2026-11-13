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

say "===== RESUME run (7B onward) ====="
SMALL=qwen-coder-1.5b-mlx
# resume: 1.5B stage and 7B speed check already done (7B: 170 s/q with SQL-only prompt, truncated reasoning at 256 tok -> 3B fallback)
LARGE=qwen-coder-3b-mlx
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
