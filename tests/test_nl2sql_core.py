from deepinsight_core.nl2sql.sql_safety import SQLSafetyChecker
from deepinsight_core.graph.nodes import _augment_retrieval_schema, _schema_columns


def test_sql_safety_allows_count_star():
    result = SQLSafetyChecker(known_tables=["orders"]).validate("SELECT COUNT(*) AS total_orders FROM orders")

    assert result.safe
    assert result.reason == ""


def test_sql_safety_rejects_top_level_select_star():
    result = SQLSafetyChecker(known_tables=["orders"]).validate("SELECT * FROM orders")

    assert not result.safe
    assert "SELECT *" in result.reason


def test_sql_safety_rejects_multiple_statements():
    result = SQLSafetyChecker(known_tables=["orders"]).validate("SELECT id FROM orders; DROP TABLE orders")

    assert not result.safe
    assert "Multiple SQL statements" in result.reason


def test_sql_safety_rejects_unknown_table():
    result = SQLSafetyChecker(known_tables=["orders"]).validate("SELECT id FROM customers")

    assert not result.safe
    assert "Unknown table" in result.reason


def test_sql_safety_rejects_unknown_qualified_column_without_sqlglot():
    sql = (
        "SELECT p.unitprice "
        "FROM products p "
        "JOIN orderdetails od ON p.ProductID = od.ProductID"
    )
    result = SQLSafetyChecker(
        known_tables=["products", "orderdetails"],
        known_columns={
            "products": ["ProductID", "Price"],
            "orderdetails": ["ProductID", "UnitPrice"],
        },
    ).validate(sql)

    assert not result.safe
    assert "Unknown column 'p.unitprice'" in result.reason


def test_schema_catalog_backfills_empty_rag_table_details():
    retrieval = {
        "database_index": ["products", "categories", "orderdetails"],
        "core_tables": ["products", "categories"],
        "core_table_details": [
            {"table_name": "products", "columns": []},
            {"table_name": "categories", "columns": []},
        ],
    }
    augmented = _augment_retrieval_schema(retrieval, {"schema_path": "data/schema_northwind.json"})
    columns = _schema_columns(augmented)

    assert "ProductID" in columns["products"]
    assert "CategoryID" in columns["categories"]
    assert "UnitPrice" in columns["orderdetails"]
