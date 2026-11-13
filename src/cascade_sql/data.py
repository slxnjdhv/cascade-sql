"""Stage `prepare`: build split manifests (one JSONL per split).

Manifest item:
    {"qid", "split", "db_id", "db_path", "question", "evidence", "difficulty",
     "gold": {"orig": sql, "arcwise": sql, "bird_official": sql, ...}}

Splits (Experiment_Design.md §4):
    g_train : random sample of BIRD train (gate fitting)
    calib   : BIRD dev MINUS the Mini-Dev question ids (Platt + thresholds + certificate)
    test_b  : BIRD Mini-Dev ids, gold under every available label version
    test_s  : Spider test (execution labels; test-suite labels imported later)
"""
import json
import os
import random
from typing import Dict, List, Optional

from .io import write_jsonl


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _bird_items(json_path: str, db_root: str, split: str) -> List[dict]:
    out = []
    for i, e in enumerate(_load_json(json_path)):
        qid = str(e.get("question_id", i))
        db = e["db_id"]
        out.append({"qid": f"bird_{qid}", "source_id": qid, "split": split, "db_id": db,
                    "db_path": os.path.join(db_root, db, f"{db}.sqlite"),
                    "question": e["question"], "evidence": e.get("evidence") or None,
                    "difficulty": e.get("difficulty"),
                    "gold": {"orig": e.get("SQL") or e.get("query")}})
    return out


def _gold_by_id(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    data = _load_json(path)
    if isinstance(data, dict):
        data = data.get("data", list(data.values()))
    return {str(e["question_id"]): (e.get("SQL") or e.get("sql") or e.get("query")) for e in data}


def prepare_bird(dev_json: str, dev_db_root: str, train_json: Optional[str], train_db_root: Optional[str],
                 minidev_ids_from: str, out_dir: str, arcwise_json: Optional[str] = None,
                 bird_official_json: Optional[str] = None, g_train_size: int = 2000, seed: int = 0,
                 g_train_from_dev: int = 0) -> dict:
    """`minidev_ids_from`: any JSON list with `question_id` (e.g. the Arcwise file) that
    defines which dev questions form Test-B."""
    dev = _bird_items(dev_json, dev_db_root, "dev")
    test_ids = set(_gold_by_id(minidev_ids_from).keys())
    arc, bofficial = _gold_by_id(arcwise_json), _gold_by_id(bird_official_json)

    calib, test_b = [], []
    for it in dev:
        if it["source_id"] in test_ids:
            it = dict(it, split="test_b")
            if it["source_id"] in arc:
                it["gold"]["arcwise"] = arc[it["source_id"]]
            if it["source_id"] in bofficial:
                it["gold"]["bird_official"] = bofficial[it["source_id"]]
            test_b.append(it)
        else:
            calib.append(dict(it, split="calib"))

    g_train = []
    if g_train_from_dev:  # laptop plan: carve G-train out of the non-test dev questions
        rng = random.Random(seed)
        idx = list(range(len(calib)))
        rng.shuffle(idx)
        take = set(idx[:g_train_from_dev])
        g_train = [dict(it, split="g_train") for i, it in enumerate(calib) if i in take]
        calib = [it for i, it in enumerate(calib) if i not in take]
    elif train_json:  # BIRD train is 8.9 GB; it may be skipped for a dev-only pilot
        train = _bird_items(train_json, train_db_root, "g_train")
        rng = random.Random(seed)
        g_train = rng.sample(train, min(g_train_size, len(train)))

    missing = test_ids - {it["source_id"] for it in test_b}
    if g_train:
        write_jsonl(os.path.join(out_dir, "g_train.jsonl"), g_train)
    write_jsonl(os.path.join(out_dir, "calib.jsonl"), calib)
    write_jsonl(os.path.join(out_dir, "test_b.jsonl"), test_b)
    bad_db = sorted({it["db_path"] for it in calib + test_b + g_train if not os.path.exists(it["db_path"])})
    report = {"g_train": len(g_train), "missing_db_files": bad_db[:5], "calib": len(calib), "test_b": len(test_b),
              "test_b_ids_not_found_in_dev": len(missing),
              "test_b_with_arcwise": sum("arcwise" in it["gold"] for it in test_b),
              "test_b_with_bird_official": sum("bird_official" in it["gold"] for it in test_b),
              "overlap_calib_test": len({i["source_id"] for i in calib} & test_ids),
              "overlap_gtrain_test": len({i["source_id"] for i in g_train} & test_ids),
              "overlap_gtrain_calib": len({i["qid"] for i in g_train} & {i["qid"] for i in calib})}
    assert report["overlap_calib_test"] == 0, "leakage: calib contains Test-B questions"
    assert report["overlap_gtrain_test"] == 0 and report["overlap_gtrain_calib"] == 0, "leakage in g_train"
    return report


def prepare_spider(test_json: str, db_root: str, out_dir: str, split: str = "test_s") -> dict:
    items = []
    for i, e in enumerate(_load_json(test_json)):
        db = e["db_id"]
        items.append({"qid": f"spider_{split}_{i}", "source_id": str(i), "split": split, "db_id": db,
                      "db_path": os.path.join(db_root, db, f"{db}.sqlite"),
                      "question": e["question"], "evidence": None, "difficulty": None,
                      "gold": {"orig": e["query"]}})
    write_jsonl(os.path.join(out_dir, f"{split}.jsonl"), items)
    return {split: len(items)}
