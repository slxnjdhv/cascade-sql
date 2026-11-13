#!/usr/bin/env bash
# Second sampling seed for the 1.5B model (the 3B escalation target is greedy, hence seed-independent).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src HF_HUB_OFFLINE=1
PY=.venv/bin/python; LOG=results/overnight.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
SMALL=qwen-coder-1.5b-mlx; LARGE=qwen-coder-3b-mlx
say "===== SEED 1 run ====="
for sp in test_b calib g_train; do
  say "generate $SMALL $sp seed1"
  $PY -m cascade_sql generate --model $SMALL --manifest data/manifests/$sp.jsonl --seed 1 >> "$LOG" 2>&1 || say "  !! generate failed"
  $PY -m cascade_sql annotate --manifest data/manifests/$sp.jsonl --records results/$SMALL/$sp.seed1.jsonl >> "$LOG" 2>&1 || say "  !! annotate failed"
done
for sp in test_b calib; do for ext in jsonl ann.jsonl; do
  [ -e results/$LARGE/$sp.seed1.$ext ] || ln -s $sp.seed0.$ext results/$LARGE/$sp.seed1.$ext; done; done
$PY -m cascade_sql fit-gate --local $SMALL --seed 1 >> "$LOG" 2>&1 || say "  !! fit-gate failed"
$PY -m cascade_sql evaluate --local $SMALL --large $LARGE --splits test_b --seed 1 >> "$LOG" 2>&1 || say "  !! evaluate failed"
say "===== SEED 1 DONE ====="
