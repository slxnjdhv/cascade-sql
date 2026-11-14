# cascade-sql

Code, data splits and analysis scripts for the paper

> Sulaxan Jadhav. **From Confidence to Decisions: Calibrated Confidence Gating for On-Device Text-to-SQL Cascades**
> (submitted to the *Journal of Intelligent Information Systems*).

A small language model running on a laptop writes several SQL candidates for a question, executes them,
and a calibrated gate turns token-level (SAC), consistency, structural and execution signals into
P(correct). The system then **answers locally**, **escalates** to a larger on-device model, or
**abstains**. A second gate decides whether to show the larger model's answer.
Everything runs on an 8 GB Apple-silicon Mac with [MLX](https://github.com/ml-explore/mlx); no GPU, no API.

The SQL-aware confidence (SAC) implementation in `src/cascade_sql/sac.py` is a port of the code released with
Maleki, Pourreza & Rafiei, *Confidence Estimation for Text-to-SQL in Large Language Models* (AAAI 2026),
under the Apache License 2.0 (see `NOTICE`).

---

## 1. Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # 16 tests
PYTHONPATH=src .venv/bin/python -m cascade_sql demo   # toy end-to-end run with mock models (~15 s)
```

Models (downloaded automatically from Hugging Face on first use, 4-bit MLX):

| role | model | size |
|---|---|---|
| small | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` | ~1 GB |
| small (ablation) | `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` | ~0.3 GB |
| escalation target | `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` | ~1.6 GB |

Generation settings are frozen in `configs/models.yaml` (prompt of Maleki et al., T = 0.7, 4 samples + greedy,
top-10 token alternatives, max 256 new tokens, one repair of the greedy query).

## 2. Data

```bash
SKIP_TRAIN=1 bash scripts/download_data.sh     # BIRD dev, Arcwise corrections, Mini-Dev
```
BIRD Mini-Dev 2025 official labels go to `data/raw/bird_minidev_2025_sqlite.json`; Spider 1.0 is unzipped
to `data/raw/spider/`. The split definitions used in the paper are **committed** in `data/manifests/`:

| split | source | n | use |
|---|---|---|---|
| `g_train` | BIRD dev (not Mini-Dev) | 600 | fit the gate |
| `calib` | BIRD dev (not Mini-Dev) | 436 | Platt calibration, threshold selection |
| `test_b` | BIRD Mini-Dev | 498 | test; labels `orig`, `bird_official` (2025), `arcwise` |
| `spider_dev` | Spider 1.0 dev | 1,034 | out-of-domain transfer |

## 3. Reproduce the paper

```bash
bash scripts/run_overnight.sh   # 1.5B on all splits, 3B escalation target, R0, gate, E1–E4, 0.5B ablation
bash scripts/run_seed1.sh       # second sampling seed for the 1.5B model
bash scripts/run_spider.sh      # Spider dev transfer
export PYTHONPATH=src
.venv/bin/python scripts/analyze_local.py --local qwen-coder-1.5b-mlx     # E1 + E6 ablation + certificate
.venv/bin/python scripts/analyze_local.py --local qwen-coder-0.5b-mlx
.venv/bin/python scripts/three_way.py --local qwen-coder-1.5b-mlx --large qwen-coder-3b-mlx   # two-stage policy
.venv/bin/python scripts/certificate_crossfit.py --local qwen-coder-1.5b-mlx                  # exchangeable control
.venv/bin/python scripts/transfer_spider.py                                                    # E5 transfer
.venv/bin/python scripts/export_errors.py --local qwen-coder-1.5b-mlx    # 50 confident-but-wrong cases (hand-labelled CSV)
.venv/bin/python scripts/draw_method.py ../paper/figures/method_overview
```
On an 8 GB M1 the full run takes about 20 hours (1.5B: ~12 s/question; 3B: ~6.5 s/question).
Runs are resumable: questions already in an output file are skipped.

Note on the 7B model: `scripts/resume_from_7b.sh` records that OmniSQL-7B was tried first and abandoned
(80–170 s/question on 8 GB; its reasoning is truncated by the 256-token budget). Its invalid outputs are not used.

## 4. Pipeline stages (CLI)

| stage | command | output |
|---|---|---|
| prepare splits | `cascade_sql prepare-bird …` / `prepare-spider …` | `data/manifests/*.jsonl` |
| generate | `cascade_sql generate --model M --manifest F [--seed S]` | `results/M/<split>.seed<S>.jsonl` |
| label + features | `cascade_sql annotate --manifest F --records R` | `….ann.jsonl` |
| base-paper reproduction | `cascade_sql reproduce --model M --split S [--label L]` | `results/eval/r0_*.md` |
| gate | `cascade_sql fit-gate --local M` | `results/gates/M.seed0/` |
| cascade evaluation | `cascade_sql evaluate --local M --large L --splits test_b` | `results/eval/M__L.seed0/report.md` + figures |

Correctness is BIRD-style execution accuracy (result sets compared as sets, floats rounded); every query
runs read-only with a timeout.

## 5. Released model outputs

All generated candidates, token log-probabilities, labels and features (`results/`, ~0.5 GB uncompressed) are archived
at Zenodo: **[doi:10.5281/zenodo.24212020](https://doi.org/10.5281/zenodo.24212020)** (CC BY 4.0). The record's
`README.md` has a short script that restores the files into `results/`, so every analysis script runs without generating anything again.

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`). BIRD, Spider and Arcwise-Plat-SQL keep their own licenses.
