#!/usr/bin/env bash
# Download the benchmark data into data/raw. Sizes: BIRD dev ~350 MB, BIRD train ~8.9 GB.
# Spider must be downloaded manually (see the note at the end).
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW"

echo "== BIRD dev (~350 MB)"
[ -f "$RAW/bird_dev.zip" ] || curl -L -o "$RAW/bird_dev.zip" https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip -n -q "$RAW/bird_dev.zip" -d "$RAW/bird_dev"
# the dev databases ship as a nested zip
find "$RAW/bird_dev" -name "dev_databases.zip" -exec unzip -n -q {} -d "$RAW/bird_dev" \;

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "== BIRD train (~8.9 GB; set SKIP_TRAIN=1 to skip)"
  [ -f "$RAW/bird_train.zip" ] || curl -L -o "$RAW/bird_train.zip" https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
  unzip -n -q "$RAW/bird_train.zip" -d "$RAW/bird_train"
  find "$RAW/bird_train" -name "train_databases.zip" -exec unzip -n -q {} -d "$RAW/bird_train" \;
fi

echo "== Arcwise-Plat-SQL corrected BIRD Mini-Dev labels (CC BY-SA 4.0)"
curl -L -o "$RAW/arcwise_plat_sql_only_with_diff.json" \
  https://raw.githubusercontent.com/uiuc-kang-lab/text_to_sql_benchmarks/main/data/arcwise_plat_sql_only_with_diff.json

echo "== BIRD Mini-Dev (official, incl. 2026 updates)"
[ -f "$RAW/minidev.zip" ] || curl -L -o "$RAW/minidev.zip" https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
unzip -n -q "$RAW/minidev.zip" -d "$RAW/minidev" || true
echo "   NOTE: the canonical updated Mini-Dev is on Hugging Face (datasets/birdsql/bird_mini_dev)."
echo "   Point --bird2026 at its SQLite JSON file when you run prepare-bird."

echo "== Spider test-suite evaluator"
[ -d "$RAW/test-suite-sql-eval" ] || git clone --depth 1 https://github.com/taoyds/test-suite-sql-eval "$RAW/test-suite-sql-eval"

cat <<'NOTE'

Spider 1.0 (incl. test set) must be downloaded manually from https://yale-lily.github.io/spider
and unzipped to data/raw/spider (it should contain test.json and test_database/).

Then look for these files (names can differ slightly between releases):
  find data/raw -name "dev.json" -o -name "train.json" -o -name "*.sqlite" | head
NOTE
