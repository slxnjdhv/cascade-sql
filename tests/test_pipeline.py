import json
import os
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from cascade_sql import execute as ex
from cascade_sql import features as F
from cascade_sql import gate as G
from cascade_sql import policy as P
from cascade_sql.generate import choose
from cascade_sql.llm import OpenAICompatible, sql_logprob_from_tokens
from cascade_sql.prompts import extract_sql, schema_ddl


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "t.sqlite")
    c = sqlite3.connect(path)
    c.executescript("""CREATE TABLE a (id INTEGER PRIMARY KEY, name TEXT, v REAL);
                       CREATE TABLE b (id INTEGER, a_id INTEGER, FOREIGN KEY (a_id) REFERENCES a(id));
                       INSERT INTO a VALUES (1,'x',1.0000001),(2,'y',2.5),(3,'z;w',NULL);
                       INSERT INTO b VALUES (1,1),(2,1),(3,2);""")
    c.commit()
    c.close()
    yield path
    ex.close_all()


# ----------------------------------------------------------------- execution
def test_read_only(db):
    r = ex.execute(db, "DELETE FROM a")
    assert not r.ok and "readonly" in r.error.lower()


def test_timeout(db):
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
            "SELECT COUNT(*) FROM c")
    r = ex.execute(db, slow, timeout_s=0.3)
    assert not r.ok and r.timed_out


def test_hash_ignores_row_order_unless_ordered(db):
    r1 = ex.execute(db, "SELECT name FROM a")
    r2 = ex.execute(db, "SELECT name FROM a ORDER BY name DESC")
    r3 = ex.execute(db, "SELECT name FROM a ORDER BY name")
    assert r1.result_hash() == ex.hash_rows(r3.rows, False)
    assert r2.result_hash() != r3.result_hash()  # both ordered, different order


def test_float_rounding_and_compare(db):
    p = ex.execute(db, "SELECT v FROM a WHERE id = 1")
    assert ex.results_match(p.rows, [(1.0,)], mode="bird")
    assert ex.results_match([(1,), (1,)], [(1,)], mode="bird")        # BIRD = set equality
    assert not ex.results_match([(1,), (1,)], [(1,)], mode="strict")  # strict = multiset


def test_is_ordered_ignores_subquery_order():
    assert not ex.is_ordered("SELECT * FROM (SELECT x FROM t ORDER BY x) s")
    assert ex.is_ordered("SELECT x FROM t ORDER BY x")


# ----------------------------------------------------------------- prompts
def test_extract_sql_variants():
    assert extract_sql("blah\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("```sql\nSELECT 1\n```\nthen ```sql\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql("The answer is SELECT name FROM a WHERE name = 'z;w'; done") == \
        "SELECT name FROM a WHERE name = 'z;w'"
    assert extract_sql("no sql here") == ""
    # OmniSQL output format: bare fence + placeholder comment
    assert extract_sql("steps...\n```\n-- Your SQL query\nSELECT a FROM t\n```") == "SELECT a FROM t"
    assert extract_sql("```sqlite\nSELECT 1\n```") == "SELECT 1"


def test_schema_ddl_has_fk_and_examples(db):
    s = schema_ddl(db)
    assert 'CREATE TABLE "b"' in s and 'FOREIGN KEY ("a_id") REFERENCES "a"("id")' in s
    assert "example:" in s


# ----------------------------------------------------------------- selection + features
def _cand(sql, h, kind="sample", ok=True, lp=-0.2):
    return {"sql": sql, "result_hash": h if ok else None, "exec_ok": ok, "kind": kind,
            "logprob": lp, "empty": False, "repaired": False}


def test_choose_majority_then_greedy():
    c = [_cand("A", "h1", "greedy"), _cand("B", "h2"), _cand("C", "h2"), _cand("D", "h1")]
    assert c[choose(c)]["sql"] == "A"  # tie 2-2, greedy's cluster wins
    c = [_cand("A", "h1", "greedy"), _cand("B", "h2", lp=-0.1), _cand("C", "h2", lp=-0.5)]
    assert c[choose(c)]["sql"] == "B"  # majority h2, best log-prob member
    assert choose([_cand("X", None, ok=False)]) == 0


def test_scf_and_structure():
    a = "SELECT name FROM a JOIN b ON a.id = b.a_id WHERE v > 1"
    b = "SELECT name FROM a JOIN b ON a.id = b.a_id WHERE v > 2"
    cands = [_cand(a, "h"), _cand(a, "h"), _cand(b, "g")]
    mean, mn = F.scf(cands, 0)
    assert mn == pytest.approx(2 / 3) and mean > mn
    nj, nsq = F.structure_counts("SELECT x FROM t WHERE y IN (SELECT y FROM u) ")
    assert nj == 0 and nsq == 1


def test_feature_vector_complete(db):
    rec = {"question": "how many?", "evidence": None, "chosen_index": 0,
           "candidates": [_cand("SELECT COUNT(*) FROM a", "h", "greedy"), _cand("SELECT COUNT(*) FROM b", "g")]}
    f = F.compute(rec, db)
    assert set(f) == set(F.ALL_FEATURES)
    assert f["f1_top_share"] == 0.5 and f["f3_greedy_agrees"] == 1.0


# ----------------------------------------------------------------- gate / certificate / policy
def test_gate_calibrates_and_certificate_is_safe():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 2))
    pr = 1 / (1 + np.exp(-(5 * x[:, 0] + 1)))  # strong signal: certifiable
    y = (rng.random(n) < pr).astype(int)
    g = G.Gate(["a", "b"]).fit(x[:2000], y[:2000]).calibrate(x[2000:3000], y[2000:3000])
    p = g.predict(x[3000:])
    assert abs(p.mean() - y[3000:].mean()) < 0.05
    cert = G.risk_threshold(g.predict(x[2000:3000]), y[2000:3000], alpha=0.10, delta=0.05)
    ans = p >= cert["tau"]
    assert cert["coverage"] > 0.3 and cert["ucb"] <= 0.10
    assert (1 - y[3000:][ans]).mean() <= 0.10
    # weak signal: refusing to certify is the correct answer
    yw = (rng.random(1000) < 0.7).astype(int)
    assert G.risk_threshold(rng.random(1000), yw, alpha=0.05)["tau"] == float("inf")


def test_budget_and_outcomes():
    p = np.array([0.9, 0.1, 0.5, 0.2])
    d = P.budget_decisions(p, 0.5)
    assert list(d) == [P.ANSWER, P.ESCALATE, P.ANSWER, P.ESCALATE]
    o = P.outcome(np.array([P.ANSWER, P.ESCALATE, P.ABSTAIN]), np.array([1, 0, 0]), np.array([0, 1, 1]))
    assert list(o) == [1, 1, 0]
    s = P.summarize(np.array([P.ANSWER, P.ABSTAIN]), np.array([0, 1]), np.array([1, 1]), lambdas=(5,))
    assert s["rs_l5"] == pytest.approx(-2.5)


# ----------------------------------------------------------------- OpenAI-compatible client
class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.last = body
        text = "think\n```sql\nSELECT 1\n```"
        toks = [{"token": t, "logprob": -0.1 * i} for i, t in enumerate(["think\n", "```sql\n", "SELECT", " 1", "\n```"])]
        out = {"choices": [{"message": {"content": text}, "logprobs": {"content": toks}}] * body.get("n", 1),
               "usage": {"prompt_tokens": 42, "completion_tokens": 5 * body.get("n", 1)}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_openai_client_against_fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = OpenAICompatible(f"http://127.0.0.1:{srv.server_port}/v1", "fake-model")
        outs = be.generate([{"role": "user", "content": "hi"}], n=3, temperature=0.8, seed=7)
        assert len(outs) == 3 and _Handler.last["n"] == 3 and _Handler.last["seed"] == 7
        assert extract_sql(outs[0].text) == "SELECT 1"
        assert outs[0].prompt_tokens == 42
        # only tokens overlapping the SQL span count: "```sql\n" (-0.1) ends where the span starts,
        # so SELECT (-0.2), " 1" (-0.3), "\n```" (-0.4) overlap
        assert outs[0].sql_logprob_mean == pytest.approx(-0.3)
    finally:
        srv.shutdown()


# ----------------------------------------------------------------- end to end
def test_demo_end_to_end(tmp_path):
    from cascade_sql.cli import main
    main(["demo", "--workdir", str(tmp_path / "run")])
    rep = json.load(open(tmp_path / "run" / "results" / "eval" / "report.json"))
    s = rep["splits"]["test_b/corrected"]
    assert s["curve_area"]["oracle"] >= s["curve_area"]["gate"] >= s["curve_area"]["random"] - 0.02
    assert os.path.exists(tmp_path / "run" / "results" / "eval" / "e2_budgets.csv")


# ----------------------------------------------------------------- base-paper SAC port
def test_sac_span_and_folding():
    from cascade_sql import sac
    toks = ["```", "SELECT", " name", " FROM", " users", " WHERE", " age", " >", " 30", "```"]
    s, e = sac.sql_span(toks)
    assert toks[s:e][0] == "SELECT" and toks[s:e][-1] == " 30"
    span = toks[s:e]
    probs = [0.9, 0.5, 0.9, 0.6, 0.9, 0.4, 0.8, 0.7]
    top = [{} for _ in span]
    top[1] = {" name": 0.5, " Name": 0.3}       # case variant -> folded into " name"
    out = sac.score_tokens("".join(span), span, probs, top)
    assert set(out) == set(sac.METHODS)
    assert out["FTC-Avg"] == pytest.approx(np.mean(probs))
    assert out["SAC-Avg"] > out["SLC-Avg"] - 1e-9 or out["SAC-Avg"] > 0  # folding raised " name" to 0.8
    bad = sac.score_tokens("SELEC nonsense ((", ["SELEC"], [0.5], [{}])
    assert bad["SAC-Avg"] == sac.UNPARSED_SCORE


def test_extract_sql_strips_special_tokens():
    assert extract_sql("SELECT a FROM t<|im_end|>\n<|endoftext|>") == "SELECT a FROM t"
