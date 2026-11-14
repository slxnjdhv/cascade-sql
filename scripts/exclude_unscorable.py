"""Mark test questions whose reference query cannot be scored reliably as unlabelled (None) in every
annotated output, so that all models and analyses use the same n.

bird_701: its reference query runs longer than the 180 s reference-execution limit on the 8 GB M1, so
whether it completed depended on machine load during annotation (it timed out for the 1.5B run but not
for the 0.5B and 3B runs). It is excluded everywhere.
usage: python scripts/exclude_unscorable.py
"""
import glob
import json
import os

EXCLUDE = {"bird_701"}
for f in sorted(glob.glob("results/*/test_b.seed*.ann.jsonl")):
    if os.path.islink(f):
        continue
    out, n = [], 0
    for line in open(f):
        r = json.loads(line)
        if r["qid"] in EXCLUDE:
            r["labels"] = {k: None for k in r.get("labels", {})}
            r["candidate_labels"] = {k: None for k in r.get("candidate_labels", {})}
            n += 1
        out.append(json.dumps(r))
    with open(f + ".tmp", "w") as g:
        g.write("\n".join(out) + "\n")
    os.replace(f + ".tmp", f)
    print(f, "excluded", n)
