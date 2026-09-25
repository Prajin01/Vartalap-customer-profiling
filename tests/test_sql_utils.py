"""Unit tests for the SQL runner (no warehouse needed)."""
from src.db import parse_named_queries, split_statements


def test_split_ignores_semicolons_in_strings_and_comments():
    sql = "SELECT 'a;b' AS x; -- comment; here\nSELECT 2;"
    assert split_statements(sql) == ["SELECT 'a;b' AS x", "SELECT 2"]


def test_parse_named_queries():
    text = "-- name: one\n-- title: First\nSELECT 1;\n\n-- name: two\nSELECT 2;\n"
    qs = parse_named_queries(text)
    assert [q["name"] for q in qs] == ["one", "two"]
    assert qs[0]["title"] == "First" and qs[0]["sql"] == "SELECT 1"
    assert qs[1]["title"] == "two"
