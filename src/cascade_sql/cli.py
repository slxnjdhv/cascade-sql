"""Command-line entry point:  python -m cascade_sql <command> [options]

Typical real run (see README.md):
  prepare-bird -> prepare-spider -> generate (local, all splits) -> generate (large, calib+tests)
  -> annotate (each file) -> fit-gate -> evaluate
"""
import argparse
import json
import os
import sys

from . import data, evaluate, generate, toy
from . import features as F
from . import labels as Lb
from .io import read_jsonl, read_yaml, write_jsonl
from .llm import make_backend


def _models(path):
    cfg = read_yaml(path)
    return cfg.get("models", {}), cfg.get("generation", {})


def cmd_generate(a):
    models, gdefault = _models(a.config)
    if a.model not in models:
        sys.exit(f"model '{a.model}' not in {a.config}; available: {list(models)}")
    mcfg = models[a.model]
    gcfg = {**gdefault, **(mcfg.get("generation") or {})}
    backend = make_backend({**mcfg, "name": a.model})
    backend.name = a.model
    items = read_jsonl(a.manifest)
    split = items[0]["split"] if items else "unknown"
    out = a.out or os.path.join(a.results, a.model, f"{split}.seed{a.seed}.jsonl")
    n = generate.run(items, backend, gcfg, out, seed=a.seed, limit=a.limit)
    print(f"generated {n} new records -> {out}")


def cmd_annotate(a):
    items = {it["qid"]: it for it in read_jsonl(a.manifest)}
    recs = read_jsonl(a.records)
    extra = {}
    for spec in a.extra or []:
        name, path = spec.split("=", 1)
        extra[name] = Lb.load_extra_labels(path)
    out, broken = [], {}
    for r in recs:
        it = items.get(r["qid"])
        if it is None:
            continue
        r = Lb.label_record(r, it, mode=a.mode, timeout_s=a.timeout)
        for name, m in extra.items():
            if r.get("qid") in m or it.get("source_id") in m:
                r["labels"][name] = m.get(r["qid"], m.get(it.get("source_id")))
        for v, val in r["labels"].items():
            if val is None:
                broken[v] = broken.get(v, 0) + 1
        r["features"] = F.compute(r, it["db_path"])
        out.append(r)
    path = a.out or a.records.replace(".jsonl", ".ann.jsonl")
    write_jsonl(path, out)
    acc = {v: round(sum(r["labels"].get(v) or 0 for r in out) / max(1, sum(r["labels"].get(v) is not None for r in out)), 4)
           for v in sorted({k for r in out for k in r["labels"]})}
    print(json.dumps({"annotated": len(out), "accuracy_by_label_version": acc,
                      "gold_sql_failed_to_execute": broken, "out": path}, indent=1))


def cmd_fit_gate(a):
    info = evaluate.fit_gate(a.results, a.local, a.seed, a.gate_dir or os.path.join(a.results, "gates", f"{a.local}.seed{a.seed}"),
                             label=a.label, learner=a.learner)
    print(json.dumps(info, indent=1))


def cmd_evaluate(a):
    gate_dir = a.gate_dir or os.path.join(a.results, "gates", f"{a.local}.seed{a.seed}")
    out = a.out or os.path.join(a.results, "eval", f"{a.local}__{a.large}.seed{a.seed}")
    evaluate.evaluate(a.results, a.local, a.large, a.seed, gate_dir, out, a.splits.split(","),
                      train_label=a.label, price_in=a.price_in, price_out=a.price_out, n_boot=a.n_boot)
    print(f"report -> {os.path.join(out, 'report.md')}")


def cmd_reproduce(a):
    rows = evaluate.reproduce(a.results, a.model, a.split, a.seed, a.out or os.path.join(a.results, "eval"), a.label)
    for r in rows:
        print(f"{r['candidate']:>7} {r['method']:<9} n={r['n']:<5} EX={r['EX']:<6} AUROC={r['AUROC']:<6} ECE={r['ECE']}")


def cmd_toy(a):
    print(json.dumps(toy.build(a.out, seed=a.seed), indent=1))


def cmd_prepare_bird(a):
    print(json.dumps(data.prepare_bird(a.dev_json, a.dev_db, a.train_json, a.train_db, a.minidev_ids,
                                       a.out, a.arcwise, a.bird_official, a.g_train_size, a.seed,
                                       g_train_from_dev=a.g_train_from_dev), indent=1))


def cmd_prepare_spider(a):
    print(json.dumps(data.prepare_spider(a.test_json, a.db_root, a.out, a.split), indent=1))


def cmd_demo(a):
    """End-to-end on the toy benchmark with mock models (no GPU, no API)."""
    root = a.workdir
    man = os.path.join(root, "data")
    res = os.path.join(root, "results")
    print("[demo] building toy benchmark:", toy.build(man, seed=0))
    cfg = {"models": {"mock-small": {"kind": "mock", "skill": 0.8, "seed": 0},
                      "mock-large": {"kind": "mock", "skill": 1.5, "seed": 1, "generation": {"n_samples": 0}}},
           "generation": {"n_samples": 8, "temperature": 0.8, "repair": True}}
    gd = cfg["generation"]
    for model, splits in (("mock-small", ["g_train", "calib", "test_b"]), ("mock-large", ["calib", "test_b"])):
        mcfg = cfg["models"][model]
        backend = make_backend({**mcfg, "name": model})
        for sp in splits:
            items = read_jsonl(os.path.join(man, f"{sp}.jsonl"))
            raw = os.path.join(res, model, f"{sp}.seed0.jsonl")
            if os.path.exists(raw):
                os.remove(raw)
            generate.run(items, backend, {**gd, **(mcfg.get("generation") or {})}, raw, seed=0, log_every=10_000)
            ns = argparse.Namespace(manifest=os.path.join(man, f"{sp}.jsonl"), records=raw, out=None,
                                    mode="bird", timeout=30.0, extra=None)
            cmd_annotate(ns)
    print(json.dumps(evaluate.fit_gate(res, "mock-small", 0, os.path.join(res, "gates", "mock-small.seed0")), indent=1)[:600])
    evaluate.evaluate(res, "mock-small", "mock-large", 0, os.path.join(res, "gates", "mock-small.seed0"),
                      os.path.join(res, "eval"), ["test_b"], price_in=1.0, price_out=4.0, n_boot=200)
    print(f"[demo] done. Open {os.path.join(res, 'eval', 'report.md')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cascade_sql")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("generate", help="run one model on one split manifest")
    s.add_argument("--config", default="configs/models.yaml")
    s.add_argument("--model", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--results", default="results")
    s.add_argument("--out")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int)
    s.set_defaults(f=cmd_generate)

    s = sub.add_parser("annotate", help="add labels (all versions) + confidence features")
    s.add_argument("--manifest", required=True)
    s.add_argument("--records", required=True)
    s.add_argument("--out")
    s.add_argument("--mode", default="bird", choices=["bird", "strict"])
    s.add_argument("--timeout", type=float, default=30.0)
    s.add_argument("--extra", action="append", help="name=path.csv with qid,label (e.g. test_suite=...)")
    s.set_defaults(f=cmd_annotate)

    s = sub.add_parser("fit-gate", help="fit gate on g_train, calibrate on calib; also fits the router baseline")
    s.add_argument("--results", default="results")
    s.add_argument("--local", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--label", default="orig")
    s.add_argument("--learner", default="logistic", choices=["logistic", "gbm"])
    s.add_argument("--gate-dir")
    s.set_defaults(f=cmd_fit_gate)

    s = sub.add_parser("evaluate", help="run E1-E4 on test splits")
    s.add_argument("--results", default="results")
    s.add_argument("--local", required=True)
    s.add_argument("--large", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--splits", default="test_b")
    s.add_argument("--label", default="orig")
    s.add_argument("--gate-dir")
    s.add_argument("--out")
    s.add_argument("--price-in", type=float, help="large model $ per 1M input tokens")
    s.add_argument("--price-out", type=float, help="large model $ per 1M output tokens")
    s.add_argument("--n-boot", type=int, default=1000)
    s.set_defaults(f=cmd_evaluate)

    s = sub.add_parser("reproduce", help="R0: base-paper protocol (FTC/SLC/SAC, AUROC, ECE)")
    s.add_argument("--results", default="results")
    s.add_argument("--model", required=True)
    s.add_argument("--split", default="calib")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--label", default="orig")
    s.add_argument("--out")
    s.set_defaults(f=cmd_reproduce)

    s = sub.add_parser("prepare-bird", help="build g_train/calib/test_b manifests from BIRD")
    s.add_argument("--dev-json", required=True)
    s.add_argument("--dev-db", required=True, help="folder containing <db_id>/<db_id>.sqlite")
    s.add_argument("--train-json", help="BIRD train.json (omit for a dev-only pilot)")
    s.add_argument("--train-db")
    s.add_argument("--minidev-ids", required=True, help="JSON list with question_id defining Test-B")
    s.add_argument("--arcwise", help="Arcwise-Plat-SQL corrected JSON")
    s.add_argument("--bird-official", help="official BIRD Mini-Dev SQLite JSON with original question_ids (archived 2025-07-04 release)")
    s.add_argument("--g-train-size", type=int, default=2000)
    s.add_argument("--g-train-from-dev", type=int, default=0,
                   help="laptop plan: take N non-test dev questions as G-train instead of BIRD train")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/manifests")
    s.set_defaults(f=cmd_prepare_bird)

    s = sub.add_parser("prepare-spider", help="build a Spider manifest")
    s.add_argument("--test-json", required=True)
    s.add_argument("--db-root", required=True)
    s.add_argument("--split", default="test_s")
    s.add_argument("--out", default="data/manifests")
    s.set_defaults(f=cmd_prepare_spider)

    s = sub.add_parser("toy", help="build the toy benchmark")
    s.add_argument("--out", default="data/toy")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(f=cmd_toy)

    s = sub.add_parser("demo", help="full pipeline on the toy benchmark with mock models")
    s.add_argument("--workdir", default="demo_run")
    s.set_defaults(f=cmd_demo)

    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
