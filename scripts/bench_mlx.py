"""Benchmark an MLX model on real BIRD questions: speed (prefill/generation tok/s, seconds per
question) and execution accuracy, for one or more prompt styles.

  PYTHONPATH=src .venv/bin/python scripts/bench_mlx.py --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
      --n 10 --styles omnisql,sqlonly
"""
import argparse
import json
import statistics as st
import time

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from cascade_sql import execute as ex
from cascade_sql import labels as L
from cascade_sql.io import read_jsonl
from cascade_sql.prompts import extract_sql, generation_messages, schema_ddl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="data/manifests/calib.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--styles", default="omnisql,sqlonly")
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()

    t0 = time.time()
    model, tok = load(a.model)
    print(f"loaded {a.model} in {time.time() - t0:.1f}s")
    items = read_jsonl(a.manifest)[: a.n]
    for style in a.styles.split(","):
        rows = []
        for it in items:
            msgs = generation_messages(schema_ddl(it["db_path"]), it["question"], it.get("evidence"), style=style)
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            t1 = time.time()
            text, last = "", None
            for r in stream_generate(model, tok, prompt, max_tokens=a.max_tokens, sampler=make_sampler(temp=0.0)):
                text += r.text
                last = r
            secs = time.time() - t1
            sql = extract_sql(text)
            ok = L.correct(it["db_path"], sql, it["gold"]["orig"], "bird")
            rows.append({"secs": secs, "ptok": last.prompt_tokens, "gtok": last.generation_tokens,
                         "ptps": last.prompt_tps, "gtps": last.generation_tps, "ok": ok or 0,
                         "peak_gb": getattr(last, "peak_memory", 0)})
        print(json.dumps({
            "style": style, "n": len(rows),
            "EX": round(sum(r["ok"] for r in rows) / len(rows), 3),
            "sec_per_q": round(st.mean(r["secs"] for r in rows), 2),
            "prompt_tok": round(st.mean(r["ptok"] for r in rows)),
            "gen_tok": round(st.mean(r["gtok"] for r in rows)),
            "prefill_tok_s": round(st.mean(r["ptps"] for r in rows)),
            "gen_tok_s": round(st.mean(r["gtps"] for r in rows), 1),
            "peak_mem_gb": round(max(r["peak_gb"] for r in rows), 2)}))
    ex.close_all()


if __name__ == "__main__":
    main()
