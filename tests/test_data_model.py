"""Integrity tests for the SQL data model (Steps 1-2)."""


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def test_staging_keeps_every_raw_row(con):
    assert scalar(con, "SELECT COUNT(*) FROM raw.transactions") == \
           scalar(con, "SELECT COUNT(*) FROM staging.lines")


def test_fact_line_primary_key_unique(con):
    assert scalar(con, "SELECT COUNT(*) - COUNT(DISTINCT line_id) FROM core.fact_line") == 0


def test_fact_invoice_primary_key_unique(con):
    assert scalar(con, "SELECT COUNT(*) - COUNT(DISTINCT invoice_no) FROM core.fact_invoice") == 0


def test_one_customer_per_invoice(con):
    n = scalar(con, """
        SELECT COUNT(*) FROM (
            SELECT invoice_no FROM core.fact_line GROUP BY invoice_no
            HAVING COUNT(DISTINCT COALESCE(customer_id, -1)) > 1)
    """)
    assert n == 0


def test_revenue_reconciles_across_layers(con):
    staging = scalar(con, "SELECT SUM(line_amount) FROM staging.lines WHERE include_in_core")
    lines = scalar(con, "SELECT SUM(line_amount) FROM core.fact_line")
    invoices = scalar(con, "SELECT SUM(invoice_amount) FROM core.fact_invoice")
    assert abs(float(staging) - float(lines)) < 0.01
    assert abs(float(lines) - float(invoices)) < 0.01


def test_fact_line_sign_rules(con):
    bad = scalar(con, """
        SELECT COUNT(*) FROM core.fact_line
        WHERE unit_price <= 0
           OR (NOT is_cancel AND quantity <= 0)
           OR (is_cancel AND quantity >= 0)
    """)
    assert bad == 0


def test_every_customer_in_dim_customer(con):
    orphans = scalar(con, """
        SELECT COUNT(DISTINCT f.customer_id)
        FROM core.fact_line f
        LEFT JOIN core.dim_customer d ON d.customer_id = f.customer_id
        WHERE f.customer_id IS NOT NULL AND d.customer_id IS NULL
    """)
    assert orphans == 0


def test_every_product_in_dim_product(con):
    orphans = scalar(con, """
        SELECT COUNT(DISTINCT f.stock_code)
        FROM core.fact_line f
        LEFT JOIN core.dim_product p ON p.stock_code = f.stock_code
        WHERE p.stock_code IS NULL
    """)
    assert orphans == 0


def test_dim_date_covers_all_invoice_dates(con):
    missing = scalar(con, """
        SELECT COUNT(DISTINCT f.invoice_date)
        FROM core.fact_line f
        LEFT JOIN core.dim_date d ON d.date_key = f.invoice_date
        WHERE d.date_key IS NULL
    """)
    assert missing == 0
