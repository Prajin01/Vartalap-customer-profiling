"""Step 1 - Ingest: convert the raw UCI Online Retail II workbook to Parquet.

Deliberately does NO cleaning. It only standardises column names and assigns a
stable raw_row_id, so every later transformation is visible in SQL.

Run:  python -m src.ingest
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from src.config import path

# Normalised header -> canonical column name. Covers both the Online Retail II
# file headers (Invoice, Price, Customer ID) and the UCI page names
# (InvoiceNo, UnitPrice, CustomerID).
COLUMN_MAP = {
    "invoice": "invoice", "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "price", "unitprice": "price",
    "customerid": "customer_id",
    "country": "country",
}
REQUIRED = ["invoice", "stock_code", "description", "quantity",
            "invoice_date", "price", "customer_id", "country"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = re.sub(r"[^a-z]", "", str(col).lower())
        if key in COLUMN_MAP:
            rename[col] = COLUMN_MAP[key]
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns {missing}; found {list(df.columns)}")
    return df[REQUIRED]


def load_workbook(xlsx_path) -> pd.DataFrame:
    """Read every sheet, keep sheet provenance, return one DataFrame."""
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    frames = []
    for sheet_name, df in sheets.items():
        df = _normalise_columns(df)
        df.insert(0, "source_sheet", str(sheet_name))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Type fixes needed only to write Parquet consistently (no rows dropped).
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df["description"] = df["description"].astype("string")
    df["country"] = df["country"].astype("string")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df.insert(0, "raw_row_id", range(1, len(df) + 1))
    return df


def main() -> None:
    src, dst = path("raw_xlsx"), path("interim_parquet")
    if not src.exists():
        sys.exit(
            f"Raw file not found: {src}\n"
            "Download 'Online Retail II' from the UCI ML Repository (dataset id 502), "
            "unzip it and place online_retail_II.xlsx in data/raw/."
        )
    print(f"Reading {src.name} (1M+ rows via openpyxl, this takes a few minutes)...")
    df = load_workbook(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    print(f"Wrote {len(df):,} rows -> {dst}")
    print(df.groupby("source_sheet").size().rename("rows").to_string())


if __name__ == "__main__":
    main()
