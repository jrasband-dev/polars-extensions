
"""Main writer module for fast SQL Server bulk inserts."""
import re
import polars as pl
import pyodbc
from typing import Optional, Literal
from tqdm.auto import tqdm
from .types import get_create_table_sql, prepare_value_for_insert

# Register as Polars DataFrame extension


@pl.api.register_dataframe_namespace("mssql")
class MSSQLNamespace:
    "Faster SQL Server Database Writing Functions"

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def write_mssql(
        self,
        table_name: str,
        connection_string: Optional[str] = None,
        connection: Optional[pyodbc.Connection] = None,
        if_exists: Literal["fail", "replace", "append", "merge"] = "fail",
        schema: str = "dbo",
        batch_size: int = 10000,
        show_progress: bool = True,
        upsert_keys: Optional[list[str]] = None,
    ) -> None:
        return write_mssql(
            self._df,
            table_name,
            connection_string=connection_string,
            connection=connection,
            if_exists=if_exists,
            schema=schema,
            batch_size=batch_size,
            show_progress=show_progress,
            upsert_keys=upsert_keys,
        )


def validate_identifier(name: str) -> str:
    # Allow only alphanumeric, underscore, and must not be empty
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


def write_mssql(
    df: pl.DataFrame,
    table_name: str,
    connection_string: Optional[str] = None,
    connection: Optional[pyodbc.Connection] = None,
    if_exists: Literal["fail", "replace", "append", "merge"] = "fail",
    schema: str = "dbo",
    batch_size: int = 10000,
    show_progress: bool = True,
    upsert_keys: Optional[list[str]] = None,
) -> None:
    """
    Write a Polars DataFrame to SQL Server with ultra-fast bulk insert or upsert.

    This function uses pyodbc's fast_executemany feature to achieve high-performance
    bulk inserts into SQL Server, typically 10-100x faster than traditional row-by-row inserts.
    It also supports upsert (MERGE) operations if requested.

    Args:
        df: Polars DataFrame to write
        table_name: Name of the target table (without schema prefix)
        connection_string: ODBC connection string (if connection not provided)
        connection: Existing pyodbc connection (if connection_string not provided)
        if_exists: How to behave if the table already exists:
            - "fail": Raise an error (default)
            - "replace": Drop the table and recreate it
            - "append": Append data to the existing table
            - "merge": Upsert data using MERGE (requires upsert_keys)
        schema: Database schema name (default: "dbo")
        batch_size: Number of rows to insert per batch (default: 10000)
        show_progress: Display progress bar during insertion (default: True)
        upsert_keys: List of column names to use as upsert keys (required for merge)

    Returns:
        None

    Raises:
        ValueError: If neither connection_string nor connection is provided
        ValueError: If table exists and if_exists="fail"
        ValueError: If if_exists="merge" and upsert_keys is not provided
        RuntimeError: If insert operation fails

    Example:
        >>> import polars as pl
        >>> from polars_mssql import write_mssql
        >>>
        >>> df = pl.DataFrame({
        ...     "id": [1, 2, 3],
        ...     "name": ["Alice", "Bob", "Charlie"],
        ...     "value": [100.5, 200.3, 300.7]
        ... })
        >>>
        >>> connection_string = (
        ...     "Driver={ODBC Driver 17 for SQL Server};"
        ...     "Server=localhost;"
        ...     "Database=mydb;"
        ...     "Trusted_Connection=yes;"
        ... )
        >>>
        >>> write_mssql(df, "my_table", connection_string, if_exists="merge", upsert_keys=["id"])
    """

    # Validate inputs
    if connection_string is None and connection is None:
        raise ValueError("Either connection_string or connection must be provided")

    if connection is None and connection_string is not None:
        should_close_connection = True
        connection = pyodbc.connect(connection_string)
    else:
        should_close_connection = False

    # Validate schema and table name
    schema = validate_identifier(schema)
    table_name = validate_identifier(table_name)

    # Validate column names
    columns = df.columns
    for col in columns:
        validate_identifier(col)

    # Defensive: never allow user input to be interpolated directly into SQL
    # All identifiers are validated above. All values use parameterized queries.

    # Validate upsert_keys if provided
    if upsert_keys is not None:
        if not isinstance(upsert_keys, list) or not all(
            isinstance(k, str) for k in upsert_keys
        ):
            raise ValueError("upsert_keys must be a list of column names (strings)")
        for k in upsert_keys:
            if k not in columns:
                raise ValueError(f"Upsert key '{k}' is not a column in the DataFrame")

    # Enable fast_executemany for bulk insert performance
    connection.autocommit = False
    cursor = connection.cursor()
    cursor.fast_executemany = True

    try:
        # Build full table name
        full_table_name = f"[{schema}].[{table_name}]"

        # Check if table exists
        table_exists_query = """
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """
        cursor.execute(table_exists_query, (schema, table_name))
        table_exists = cursor.fetchone()[0] > 0

        # Handle table existence based on if_exists parameter
        if table_exists:
            if if_exists == "fail":
                raise ValueError(
                    f"Table {full_table_name} already exists. "
                    f"Use if_exists='replace', 'append', or 'merge' to modify this behavior."
                )
            elif if_exists == "replace":
                # Drop the existing table
                cursor.execute(f"DROP TABLE {full_table_name}")
                table_exists = False
            elif if_exists == "merge":
                # Always use upsert logic if merge is selected
                if upsert_keys is None or not upsert_keys:
                    raise ValueError(
                        "if_exists='merge' requires upsert_keys to be provided."
                    )
                use_upsert = True

        # Create table if it doesn't exist
        if not table_exists:
            create_sql = get_create_table_sql(table_name, schema, df)
            cursor.execute(create_sql)

        # Prepare data for insertion
        num_rows = len(df)

        if num_rows == 0:
            # No data to insert
            connection.commit()
            return

        columns = df.columns
        columns_str = ", ".join([f"[{col}]" for col in columns])
        placeholders = ", ".join(["?"] * len(columns))
        insert_sql = (
            f"INSERT INTO {full_table_name} ({columns_str}) VALUES ({placeholders})"
        )

        # Upsert support: if upsert_keys is provided, use MERGE
        use_upsert = upsert_keys is not None and len(upsert_keys) > 0
        if use_upsert:
            # All upsert_keys and columns are validated above
            on_clause = " AND ".join(
                [f"target.[{k}] = source.[{k}]" for k in upsert_keys]
            )
            update_cols = [col for col in columns if col not in upsert_keys]
            update_set = ", ".join(
                [f"target.[{col}] = source.[{col}]" for col in update_cols]
            )
            insert_cols = ", ".join([f"[{col}]" for col in columns])
            insert_vals = ", ".join([f"source.[{col}]" for col in columns])

        # Process in batches to manage memory
        num_batches = (num_rows + batch_size - 1) // batch_size
        batch_iterator = range(0, num_rows, batch_size)
        if show_progress:
            batch_iterator = tqdm(
                batch_iterator,
                desc=f"Writing to {table_name}",
                unit="batch",
                total=num_batches,
                unit_scale=False,
            )

        for batch_start in batch_iterator:
            batch_end = min(batch_start + batch_size, num_rows)
            batch_df = df[batch_start:batch_end]

            # Convert to row-oriented format
            rows = []
            for row_dict in batch_df.iter_rows(named=False):
                prepared_row = tuple(
                    prepare_value_for_insert(val, dtype)
                    for val, dtype in zip(row_dict, df.dtypes)
                )
                rows.append(prepared_row)

            if use_upsert:
                # Defensive: temp table name is validated
                temp_table = f"#temp_upsert_{table_name}"
                temp_table = validate_identifier(temp_table.replace("#", ""))
                temp_table = f"#{temp_table}"
                # Drop temp table if exists (safe, validated name)
                cursor.execute(
                    f"IF OBJECT_ID('tempdb..[{temp_table[1:]}]') IS NOT NULL DROP TABLE {temp_table}"
                )
                # Create temp table with same structure
                create_temp_sql = get_create_table_sql(temp_table, None, batch_df)
                cursor.execute(create_temp_sql)
                # Insert batch into temp table
                temp_insert_sql = (
                    f"INSERT INTO {temp_table} ({columns_str}) VALUES ({placeholders})"
                )
                cursor.fast_executemany = True
                cursor.executemany(temp_insert_sql, rows)
                # Build and execute MERGE statement
                # All identifiers are validated, values are parameterized
                merge_sql = (
                    f"MERGE INTO {full_table_name} AS target "
                    f"USING {temp_table} AS source "
                    f"ON {on_clause} "
                    f"WHEN MATCHED THEN UPDATE SET {update_set} "
                    f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals});"
                )
                cursor.execute(merge_sql)
                # Drop temp table
                cursor.execute(f"DROP TABLE {temp_table}")
            else:
                # Execute bulk insert for this batch
                cursor.executemany(insert_sql, rows)

            # Update progress bar with row count if enabled
            if show_progress and hasattr(batch_iterator, "set_postfix"):
                batch_iterator.set_postfix(rows=f"{batch_end}/{num_rows}")

        # Commit the transaction
        connection.commit()

    except Exception as e:
        # Rollback on error
        connection.rollback()
        raise RuntimeError(f"Failed to write data to SQL Server: {str(e)}") from e

    finally:
        cursor.close()
        if should_close_connection:
            connection.close()
