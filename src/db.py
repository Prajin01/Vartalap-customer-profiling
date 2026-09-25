"""DuckDB helpers: connect, run SQL files, run named queries.

All data transformation lives in sql/. Python only orchestrates it.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from src.config import path


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db_file = path("duckdb")
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_file), read_only=read_only)


def strip_comments(sql: str) -> str:
    """Remove `-- ...` comments that are outside single-quoted strings."""
    out, i, in_str = [], 0, False
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            in_str = not in_str
            out.append(ch)
            i += 1
        elif not in_str and sql.startswith("--", i):
            while i < len(sql) and sql[i] != "\n":
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def split_statements(sql: str) -> list[str]:
    """Split a SQL script on semicolons that are outside single-quoted strings."""
    sql = strip_comments(sql)
    stmts, buf, in_str = [], [], False
    for ch in sql:
        if ch == "'":
            in_str = not in_str
        if ch == ";" and not in_str:
            stmt = "".join(buf).strip()
            if stmt:
                stmts.append(stmt)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


def run_sql_file(con: duckdb.DuckDBPyConnection, sql_file: Path) -> None:
    """Execute every statement in a .sql file, in order."""
    for stmt in split_statements(Path(sql_file).read_text(encoding="utf-8")):
        con.execute(stmt)


_NAME_RE = re.compile(r"^--\s*name:\s*(\S+)\s*$", re.MULTILINE)
_TITLE_RE = re.compile(r"^--\s*title:\s*(.+?)\s*$", re.MULTILINE)


def parse_named_queries(sql_text: str) -> list[dict]:
    """Parse a file of blocks that start with `-- name: <id>` (optional `-- title: ...`)."""
    matches = list(_NAME_RE.finditer(sql_text))
    queries = []
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(sql_text)
        block = sql_text[m.end():end]
        title_m = _TITLE_RE.search(block)
        stmts = split_statements(block)
        if not stmts:
            continue
        queries.append({
            "name": m.group(1),
            "title": title_m.group(1) if title_m else m.group(1),
            "sql": stmts[-1],
        })
    return queries


def run_named_queries(con: duckdb.DuckDBPyConnection, sql_file: Path) -> list[dict]:
    """Run each named query and attach its result as a DataFrame under key 'df'."""
    queries = parse_named_queries(Path(sql_file).read_text(encoding="utf-8"))
    for q in queries:
        q["df"] = con.execute(q["sql"]).df()
    return queries


def query_df(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    return con.execute(sql, params or []).df()
