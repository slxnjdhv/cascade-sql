"""Logit-based confidence for generated SQL: FTC, SLC and SAC (Avg / Prod).

Ported from the base paper's reference implementation:
    Maleki, Pourreza & Rafiei, "Confidence Estimation for Text-to-SQL in Large Language
    Models", AAAI 2026. https://github.com/sepidehntzr/Text2SQL-Confidence (confidence.py)
    Copyright the original authors, licensed under the Apache License 2.0
    (http://www.apache.org/licenses/LICENSE-2.0).
Changes from the original: packaged as functions over our candidate records; the token
span selection from generate_sql.py (`sql_span`) is included here; ECE lives in metrics.py.
The scoring logic itself is unchanged, so results are comparable with the base paper.
"""
import string
from typing import Dict, List

import numpy as np
import sqlglot
from sqlglot import exp

SQL_KEYWORDS = {
    "select", "from", "where", "join", "inner", "left", "right", "outer", "full", "cross",
    "on", "as", "and", "or", "not", "in", "is", "null", "like", "between", "exists",
    "group", "by", "order", "having", "limit", "offset", "asc", "desc", "distinct",
    "union", "intersect", "except", "all", "case", "when", "then", "else", "end",
    "cast", "real", "integer", "count", "sum", "avg", "min", "max", "abs", "round",
    "length", "substr", "instr", "strftime", "iif",
}
SAC_SKIP = {"select", "from", "as", "inner", "outer", "on", "by"}
OPERATORS = {"=", "==", "!=", "<>", "<", ">", "<=", ">=", "+", "-", "*", "/", "%", "||"}
SYNONYMS = [{"!=", "<>"}, {"=", "=="}]
UNPARSED_SCORE = 0.5
METHODS = [f"{m}-{a}" for m in ("FTC", "SLC", "SAC") for a in ("Avg", "Prod")]


def sql_span(tokens: List[str]):
    """Start/end index of the query: from SELECT up to ';' or the closing ``` (base paper)."""
    stripped = [t.strip() for t in tokens]
    start = stripped.index("SELECT") if "SELECT" in stripped else 0
    end = len(tokens)
    for i in range(start, len(tokens)):
        if stripped[i].startswith(";"):
            end = i + 1
            break
        if stripped[i].startswith("```"):
            end = i
            break
    return start, end


def norm(tok):
    return tok.strip().strip("`\"'[]").lower()


def is_punct(t):
    return all(c in string.punctuation or c.isspace() for c in t)


def schema_items(tree):
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    columns = {c.name.lower() for c in tree.find_all(exp.Column)}
    values = {str(v.this).lower() for v in tree.find_all(exp.Literal)}
    return tables | columns | values


def in_schema(t, items):
    return t in items or (len(t) > 1 and any(t in item for item in items))


def interchangeable_groups(tree):
    groups = []
    for select in tree.find_all(exp.Select):
        groups.append({c.name.lower() for e in select.expressions for c in e.find_all(exp.Column)})
        group_by = select.args.get("group")
        if group_by is not None:
            groups.append({c.name.lower() for c in group_by.find_all(exp.Column)})
        joins = select.args.get("joins") or []
        if all(not j.side for j in joins):
            groups.append({t.name.lower() for t in select.find_all(exp.Table)
                           if t.find_ancestor(exp.Select) is select})
        where = select.args.get("where")
        if where is not None:
            conds = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
            firsts = {next(c.find_all(exp.Column)).name.lower()
                      for c in conds if next(c.find_all(exp.Column), None) is not None}
            groups.append(firsts)
    for eq in tree.find_all(exp.EQ):
        if isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
            groups.append({eq.left.name.lower(), eq.right.name.lower()})
    return [g for g in groups if len(g) > 1]


def fold(token, prob, alternatives, groups):
    t = norm(token)
    swappable = set()
    for g in groups:
        if any(name.startswith(t) for name in g):
            swappable |= {name for name in g if not name.startswith(t)}
    synonyms = next((s for s in SYNONYMS if t in s), set())
    extra = 0.0
    for alt, p in alternatives.items():
        if alt == token:
            continue
        a = norm(alt)
        if not a:
            continue
        if a == t or a in synonyms or (len(a) > 1 and any(n.startswith(a) for n in swappable)):
            extra += p
    return min(1.0, prob + extra)


def slc_scores(tokens, probs, items):
    scores = []
    for tok, p in zip(tokens, probs):
        t = norm(tok)
        if t and not is_punct(t) and t not in SQL_KEYWORDS and in_schema(t, items):
            scores.append(p)
    return scores


def sac_scores(tokens, probs, top, items, groups):
    scores = []
    for tok, p, alts in zip(tokens, probs, top):
        t = norm(tok)
        if not t or t in SAC_SKIP:
            continue
        if is_punct(t) and t not in OPERATORS:
            continue
        if t in SQL_KEYWORDS or t in OPERATORS or in_schema(t, items):
            scores.append(fold(tok, p, alts, groups))
    return scores


def aggregate(scores, how):
    if not scores:
        return UNPARSED_SCORE
    return float(np.mean(scores)) if how == "avg" else float(np.prod(scores))


def score_tokens(sql: str, tokens: List[str], probs: List[float], top: List[Dict[str, float]]) -> Dict[str, float]:
    """FTC/SLC/SAC × Avg/Prod for one query, given its SQL-span tokens."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        tree = None
    if tree is None or not tokens:
        return {m: UNPARSED_SCORE for m in METHODS}
    items = schema_items(tree)
    per = {"FTC": probs, "SLC": slc_scores(tokens, probs, items),
           "SAC": sac_scores(tokens, probs, top, items, interchangeable_groups(tree))}
    out = {}
    for name, scores in per.items():
        out[f"{name}-Avg"] = aggregate(scores, "avg")
        out[f"{name}-Prod"] = aggregate(scores, "prod")
    return out


def score_candidate(cand: dict) -> Dict[str, float]:
    """Scores for one of our candidate records (needs cand['token_data'] from the MLX backend)."""
    td = cand.get("token_data")
    if not td:
        return {m: float("nan") for m in METHODS}
    tokens = [t["token"] for t in td]
    s, e = sql_span(tokens)
    td = td[s:e]
    tokens = [t["token"] for t in td]
    probs = [float(np.exp(t["logprob"])) for t in td]
    top = [t.get("top", {}) for t in td]
    sql = "".join(tokens).strip()
    return score_tokens(sql, tokens, probs, top)
