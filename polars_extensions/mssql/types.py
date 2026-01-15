"""Type mapping utilities for converting Polars types to SQL Server types."""

import polars as pl


def polars_to_sqlserver_type(dtype: pl.DataType, column_name: str = "") -> str:
    """
    Convert a Polars data type to SQL Server data type.

    Args:
        dtype: Polars data type
        column_name: Column name (for error messages)

    Returns:
        SQL Server type string

    Raises:
        ValueError: If the data type is not supported
    """
    # Integer types
    if dtype == pl.Int8 or dtype == pl.Int16:
        return "SMALLINT"
    elif dtype == pl.Int32:
        return "INT"
    elif dtype == pl.Int64:
        return "BIGINT"

    # Unsigned integer types (map to signed equivalents)
    elif dtype == pl.UInt8 or dtype == pl.UInt16 or dtype == pl.UInt32:
        return "INT"
    elif dtype == pl.UInt64:
        return "BIGINT"

    # Float types
    elif dtype == pl.Float32:
        return "REAL"
    elif dtype == pl.Float64:
        return "FLOAT"

    # Boolean
    elif dtype == pl.Boolean:
        return "BIT"

    # String types
    elif dtype == pl.Utf8 or dtype == pl.Categorical or dtype == pl.String:
        return "NVARCHAR(MAX)"

    # Date and time types
    elif dtype == pl.Date:
        return "DATE"
    elif dtype == pl.Datetime:
        return "DATETIME2"
    elif dtype == pl.Time:
        return "TIME"
    elif dtype == pl.Duration:
        return "BIGINT"  # Store as microseconds

    # Decimal
    elif isinstance(dtype, pl.Decimal):
        # Use precision and scale if available
        return "DECIMAL(38, 10)"

    # Binary
    elif dtype == pl.Binary:
        return "VARBINARY(MAX)"

    # Null type - default to NVARCHAR
    elif dtype == pl.Null:
        return "NVARCHAR(MAX)"

    else:
        raise ValueError(
            f"Unsupported Polars data type for column '{column_name}': {dtype}. "
            f"Please convert the column to a supported type before writing to SQL Server."
        )


def get_create_table_sql(table_name: str, schema: str, df: pl.DataFrame) -> str:
    """
    Generate CREATE TABLE SQL statement from a Polars DataFrame.

    Args:
        table_name: Name of the table to create
        schema: Database schema name
        df: Polars DataFrame

    Returns:
        SQL CREATE TABLE statement
    """
    columns = []
    for col_name in df.columns:
        col_type = df[col_name].dtype
        sql_type = polars_to_sqlserver_type(col_type, col_name)
        # Escape column names with brackets
        columns.append(f"[{col_name}] {sql_type}")

    columns_sql = ",\n    ".join(columns)

    return f"""CREATE TABLE [{schema}].[{table_name}] (
    {columns_sql}
)"""


def prepare_value_for_insert(value, dtype: pl.DataType):
    """
    Prepare a single value for SQL Server insertion.

    Args:
        value: The value to prepare
        dtype: Polars data type

    Returns:
        Prepared value suitable for pyodbc
    """
    if value is None:
        return None

    # Duration types - convert to microseconds
    if dtype == pl.Duration:
        if hasattr(value, "total_seconds"):
            return int(value.total_seconds() * 1_000_000)
        return value

    # Date/datetime - ensure proper format
    if dtype == pl.Date or dtype == pl.Datetime:
        return value

    # Boolean - convert to 0/1
    if dtype == pl.Boolean:
        return 1 if value else 0

    return value
