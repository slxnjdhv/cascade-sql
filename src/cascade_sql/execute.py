"""Sandboxed SQLite execution and result comparison.

* Databases are opened read-only (URI mode=ro), so generated SQL cannot modify them.
* A progress handler aborts queries that run past the timeout.
* Results are normalised (floats rounded, rows sorted unless the query is ordered)
  and hashed, so candidates can be clustered by what they return.
"""
import hashlib
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

MAX_ROWS = 10_000          # rows fetched per query (beyond this the result is marked truncated)
FLOAT_DIGITS = 6


@dataclass
class ExecResult:
    ok: bool
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    error: Optional[str] = None
    timed_out: bool = False
    truncated: bool = False
    seconds: float = 0.0
    ordered: bool = False

    @property
    def empty(self) -> bool:
        return self.ok and (len(self.rows) == 0 or all(all(v is None for v in r) for r in self.rows))

    def result_hash(self) -> Optional[str]:
        if not self.ok:
            return None
        return hash_rows(self.rows, self.ordered)


_conn_cache = {}


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Read-only, immutable connection (benchmark DBs never change). `immutable=1` avoids
    lock/journal files, which also makes large BIRD databases work in sandboxes and on
    read-only mounts. Paths with spaces are percent-encoded via as_uri()."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    uri = Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def _connect(db_path: str) -> sqlite3.Connection:
    conn = _conn_cache.get(db_path)
    if conn is None:
        conn = open_readonly(db_path)
        _conn_cache[db_path] = conn
    return conn


def close_all() -> None:
    for c in _conn_cache.values():
        try:
            c.close()
        except Exception:
            pass
    _conn_cache.clear()


_ORDER_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def is_ordered(sql: str) -> bool:
    """True if the *outermost* query has ORDER BY (crude but standard: last ORDER BY
    appears after the last closing parenthesis)."""
    m = list(_ORDER_RE.finditer(sql))
    if not m:
        return False
    return m[-1].start() > sql.rfind(")")


def execute(db_path: str, sql: str, timeout_s: float = 30.0) -> ExecResult:
    if not sql or not sql.strip():
        return ExecResult(ok=False, error="empty SQL")
    conn = _connect(db_path)
    deadline = time.monotonic() + timeout_s
    state = {"timed_out": False}

    def _handler():
        if time.monotonic() > deadline:
            state["timed_out"] = True
            return 1  # non-zero aborts the query
        return 0

    conn.set_progress_handler(_handler, 10_000)
    t0 = time.monotonic()
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS + 1)
        truncated = len(rows) > MAX_ROWS
        rows = [tuple(r) for r in rows[:MAX_ROWS]]
        return ExecResult(ok=True, rows=rows, truncated=truncated,
                          seconds=time.monotonic() - t0, ordered=is_ordered(sql))
    except Exception as e:  # sqlite3.Error, Warning (multiple statements), etc.
        msg = "query timed out" if state["timed_out"] else f"{type(e).__name__}: {e}"
        return ExecResult(ok=False, error=msg, timed_out=state["timed_out"],
                          seconds=time.monotonic() - t0)
    finally:
        conn.set_progress_handler(None, 0)


def _norm_value(v: Any) -> Any:
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        r = round(v, FLOAT_DIGITS)
        return int(r) if r.is_integer() else r
    if isinstance(v, bytes):
        return v.hex()
    return v


def _norm_rows(rows):
    return [tuple(_norm_value(v) for v in r) for r in rows]


def hash_rows(rows, ordered: bool = False) -> str:
    nr = _norm_rows(rows)
    if not ordered:
        nr = sorted(nr, key=repr)
    return hashlib.sha1(repr(nr).encode("utf-8")).hexdigest()[:16]


def results_match(pred_rows, gold_rows, mode: str = "bird", gold_ordered: bool = False) -> bool:
    """Compare two result sets.

    mode="bird":   set equality, exactly what the official BIRD evaluator does.
    mode="strict": multiset equality; order also enforced when the gold query is ordered
                   (closer to the Spider execution evaluator).
    """
    p, g = _norm_rows(pred_rows), _norm_rows(gold_rows)
    if mode == "bird":
        return set(p) == set(g)
    if gold_ordered:
        return p == g
    return sorted(p, key=repr) == sorted(g, key=repr)
