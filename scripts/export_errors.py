"""Export the N most confident-but-wrong test_b answers of the hybrid gate for manual error analysis.
usage: PYTHONPATH=src python scripts/export_errors.py --local qwen-coder-1.5b-mlx [--n 50] [--label orig]"""
import argparse, csv, json, os
import numpy as np
from cascade_sql.evaluate import load, matrix
from cascade_sql.features import ALL_FEATURES
from cascade_sql.gate import Gate

ap = argparse.ArgumentParser()
ap.add_argument("--local", required=True); ap.add_argument("--n", type=int, default=50)
ap.add_argument("--label", default="orig"); ap.add_argument("--results", default="results")
a = ap.parse_args()
gate = Gate.load(os.path.join(a.results, "gates", f"{a.local}.seed0", "gate.pkl"))
recs = list(load(a.results, a.local, "test_b", 0).values())
X, y, _, keep = matrix(recs, ALL_FEATURES, a.label)
recs = [r for r, k in zip(recs, keep) if k]
p = gate.predict(X)
gold = {json.loads(l)["qid"]: json.loads(l) for l in open("data/manifests/test_b.jsonl")}
order = [i for i in np.argsort(-p) if y[i] == 0][: a.n]
os.makedirs(os.path.join(a.results, "analysis"), exist_ok=True)
out = os.path.join(a.results, "analysis", f"confident_wrong_{a.local}_{a.label}.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "qid", "db_id", "p_gate", "top_share", "question", "evidence", "pred_sql", "gold_sql",
                "difficulty", "error_category (fill in)", "notes"])
    for k, i in enumerate(order, 1):
        r = recs[i]; g = gold.get(r["qid"], {})
        w.writerow([k, r["qid"], r["db_id"], round(float(p[i]), 3), r["features"].get("f1_top_share"),
                    r["question"], r.get("evidence"), r["chosen_sql"], (g.get("gold") or {}).get(a.label), g.get("difficulty"), "", ""])
print(out, len(order))
