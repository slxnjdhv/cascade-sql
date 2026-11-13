"""Schema serialisation, prompt templates and SQL extraction.

The template follows the general shape used by OmniSQL-style SQL models
(DDL schema + question + external knowledge, answer in a ```sql block).
Before the real runs, compare it with the official template on the model card of
each model you use and paste that in here if it differs. Keep ONE template per
model, and freeze it before E1.
"""
import re
import sqlite3
from functools import lru_cache
from typing import Dict, List, Optional

from .execute import open_readonly


@lru_cache(maxsize=512)
def schema_ddl(db_path: str, n_samples: int = 3) -> str:
    """CREATE TABLE statements with PK/FK info and a few example values per column."""
    conn = open_readonly(db_path)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        parts = []
        for t in tables:
            cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()  # cid,name,type,notnull,dflt,pk
            fks = conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall()  # id,seq,table,from,to,...
            lines = []
            for _, name, ctype, _, _, pk in cols:
                ex = ""
                if n_samples > 0:
                    try:
                        vals = [r[0] for r in conn.execute(
                            f'SELECT DISTINCT "{name}" FROM "{t}" WHERE "{name}" IS NOT NULL LIMIT {n_samples}')]
                        vals = [str(v)[:40] for v in vals]
                        if vals:
                            ex = "  -- example: " + ", ".join(repr(v) for v in vals)
                    except sqlite3.Error:
                        pass
                lines.append(f'  "{name}" {ctype or ""}{" PRIMARY KEY" if pk == 1 else ""},{ex}')
            pks = [c[1] for c in cols if c[5] > 0]
            if len(pks) > 1:
                lines.append("  PRIMARY KEY (" + ", ".join(f'"{p}"' for p in pks) + "),")
            for fk in fks:
                lines.append(f'  FOREIGN KEY ("{fk[3]}") REFERENCES "{fk[2]}"("{fk[4]}"),')
            if lines:
                lines[-1] = lines[-1].replace(",  --", "  --").rstrip(",")
            parts.append(f'CREATE TABLE "{t}" (\n' + "\n".join(lines) + "\n);")
        return "\n\n".join(parts)
    finally:
        conn.close()


@lru_cache(maxsize=512)
def schema_stats(db_path: str) -> Dict[str, int]:
    conn = open_readonly(db_path)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        n_cols = sum(len(conn.execute(f'PRAGMA table_info("{t}")').fetchall()) for t in tables)
        return {"n_tables": len(tables), "n_columns": n_cols}
    finally:
        conn.close()


SYSTEM = ("You are a careful SQLite expert. Write one SQLite query that answers the question "
          "using only the given schema.")

GEN_TEMPLATE = """Database schema:
{schema}

{evidence_block}Question: {question}

Think briefly about which tables and columns are needed, then give the final SQLite query in a ```sql code block."""

REPAIR_TEMPLATE = """Database schema:
{schema}

{evidence_block}Question: {question}

This SQLite query failed:
```sql
{sql}
```
Error: {error}

Fix the query. Give only the corrected SQLite query in a ```sql code block."""


def _evidence_block(evidence: Optional[str]) -> str:
    return f"External knowledge: {evidence}\n\n" if evidence else ""


# Official OmniSQL template (github.com/RUCKBReasoning/OmniSQL, Apache-2.0). External knowledge
# is appended to the question, as the OmniSQL README instructs. Single user message, no system.
OMNISQL_TEMPLATE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""

OMNISQL_REPAIR_SUFFIX = """

A previous attempt produced this query, which failed when executed:
```
{sql}
```
Error: {error}

Write a corrected query, following the same output format."""

PROMPT_STYLES = ("maleki", "omnisql", "generic", "sqlonly")

# Base-paper prompt (Maleki et al., AAAI 2026; github.com/sepidehntzr/Text2SQL-Confidence,
# generate_sql.py, Apache-2.0): CREATE statements + 3 rows per table, one-line SQL output.
MALEKI_INSTRUCTION = """Generate a SQLite query that answers the question, using the database schema and the evidence below.
Write the query in one line between ``` and ```, with no explanation.
Use table and column names exactly as they appear in the schema."""


@lru_cache(maxsize=512)
def schema_with_rows(db_path: str, n_rows: int = 3) -> str:
    """Base-paper schema format: each CREATE TABLE statement followed by a few rows."""
    conn = open_readonly(db_path)
    try:
        cur = conn.cursor()
        tables = cur.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        parts = []
        for name, create_sql in tables:
            rows = cur.execute(f'SELECT * FROM "{name}" LIMIT {n_rows}').fetchall()
            header = " | ".join(d[0] for d in cur.description)
            body = "\n".join(" | ".join(str(v) for v in r) for r in rows)
            parts.append(f"{create_sql}\n/*\n{n_rows} rows from {name}:\n{header}\n{body}\n*/")
        return "\n\n".join(parts)
    finally:
        conn.close()


def maleki_prompt(schema: str, question: str, evidence: Optional[str]) -> str:
    prompt = f"{MALEKI_INSTRUCTION}\n\nSchema:\n{schema}\n\nQuestion: {question}\n"
    if evidence:
        prompt += f"Evidence: {evidence}\n"
    return prompt

# Short-output style for CPU/laptop runs: no step-by-step reasoning, so ~5-10x fewer output tokens.
SQLONLY_TEMPLATE = """Database schema:
{schema}

{evidence_block}Question: {question}

Return only the SQLite query in a ```sql code block, with no explanation."""


def _omni_question(question: str, evidence: Optional[str]) -> str:
    return f"{evidence}\n{question}" if evidence else question


def generation_messages(schema: str, question: str, evidence: Optional[str],
                        style: str = "omnisql") -> List[Dict[str, str]]:
    if style == "maleki":  # `schema` must come from schema_with_rows()
        return [{"role": "user", "content": maleki_prompt(schema, question, evidence)}]
    if style == "omnisql":
        return [{"role": "user", "content": OMNISQL_TEMPLATE.format(
            db_details=schema, question=_omni_question(question, evidence))}]
    if style == "generic":
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": GEN_TEMPLATE.format(
                    schema=schema, question=question, evidence_block=_evidence_block(evidence))}]
    if style == "sqlonly":
        return [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": SQLONLY_TEMPLATE.format(
                    schema=schema, question=question, evidence_block=_evidence_block(evidence))}]
    raise ValueError(f"prompt style must be one of {PROMPT_STYLES}")


def repair_messages(schema: str, question: str, evidence: Optional[str], sql: str, error: str,
                    style: str = "omnisql"):
    if style == "maleki":
        return [{"role": "user", "content": maleki_prompt(schema, question, evidence)
                 + f"\nA previous query failed when executed:\n```{sql}```\nError: {error[:500]}\n"
                 + "Write a corrected query in one line between ``` and ```, with no explanation.\n"}]
    if style == "omnisql":
        base = OMNISQL_TEMPLATE.format(db_details=schema, question=_omni_question(question, evidence))
        return [{"role": "user", "content": base + OMNISQL_REPAIR_SUFFIX.format(sql=sql, error=error[:500])}]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": REPAIR_TEMPLATE.format(
                schema=schema, question=question, evidence_block=_evidence_block(evidence),
                sql=sql, error=error[:500])}]


def is_repair_prompt(messages: List[Dict[str, str]]) -> bool:
    return "failed" in messages[-1]["content"]


_FENCE_SQL = re.compile(r"```(?:sqlite|sql)\b\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_FENCE_ANY = re.compile(r"```\s*(.*?)```", re.DOTALL)
_SELECT = re.compile(r"\b(WITH|SELECT)\b.*", re.IGNORECASE | re.DOTALL)


_SPECIAL = re.compile(r"<\|[^|<>]{1,40}\|>")


def extract_sql(text: str) -> str:
    """Take the last ```sql block; fall back to any code block, then to the first SELECT/WITH."""
    if not text:
        return ""
    text = _SPECIAL.sub("", text)  # never let chat-template markers like <|im_end|> into SQL
    for rx in (_FENCE_SQL, _FENCE_ANY):
        m = rx.findall(text)
        if m:
            return _clean(m[-1])
    m = _SELECT.search(text)
    return _clean(m.group(0)) if m else ""


def _clean(sql: str) -> str:
    sql = sql.strip()
    # drop a language tag left on the first line of a bare ``` fence (e.g. "sqlite")
    first, _, rest = sql.partition("\n")
    if rest and first.strip().lower() in ("sql", "sqlite"):
        sql = rest.strip()
    # drop whole-line "--" comments (e.g. OmniSQL's "-- Your SQL query" placeholder)
    sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--")).strip()
    # keep only the first statement (quote-aware, so 'a;b' inside a string survives)
    quote = None
    for i, ch in enumerate(sql):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == ";":
            sql = sql[:i]
            break
    return sql.strip()


def approx_tokens(text: str) -> int:
    """Rough token count (≈4 chars/token). Real counts come from the API `usage` field."""
    return max(1, len(text) // 4)
