#!/usr/bin/env bash
# Out-of-domain transfer (E5): run the 1.5B and 3B models on Spider dev (1,034 questions, 20 databases).
# Waits for the seed-1 run to finish first (one model in memory at a time on 8 GB).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src HF_HUB_OFFLINE=1
PY=.venv/bin/python; LOG=results/overnight.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while pgrep -f run_seed1.sh >/dev/null; do sleep 60; done
say "===== SPIDER run ====="
for m in qwen-coder-1.5b-mlx qwen-coder-3b-mlx; do
  say "generate $m spider_dev"
  $PY -m cascade_sql generate --model $m --manifest data/manifests/spider_dev.jsonl >> "$LOG" 2>&1 || say "  !! generate failed"
  $PY -m cascade_sql annotate --manifest data/manifests/spider_dev.jsonl --records results/$m/spider_dev.seed0.jsonl >> "$LOG" 2>&1 || say "  !! annotate failed"
done
say "===== SPIDER DONE ====="
