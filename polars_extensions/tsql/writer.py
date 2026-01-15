"""Main writer module for fast SQL Server bulk inserts."""

import polars as pl
import pyodbc
from typing import Union, Optional, Literal
from tqdm.auto import tqdm
from .types import get_create_table_sql, prepare_value_for_insert

# Register as Polars DataFrame extension
import polars as pl


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
        if_exists: Literal["fail", "replace", "append"] = "fail",
        schema: str = "dbo",
        batch_size: int = 10000,
        show_progress: bool = True,
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
        )


def write_mssql(
    df: pl.DataFrame,
    table_name: str,
    connection_string: Optional[str] = None,
    connection: Optional[pyodbc.Connection] = None,
    if_exists: Literal["fail", "replace", "append"] = "fail",
    schema: str = "dbo",
    batch_size: int = 10000,
    show_progress: bool = True,
) -> None:
    """
    Write a Polars DataFrame to SQL Server with ultra-fast bulk insert.

    This function uses pyodbc's fast_executemany feature to achieve high-performance
    bulk inserts into SQL Server, typically 10-100x faster than traditional row-by-row inserts.

    Args:
        df: Polars DataFrame to write
        table_name: Name of the target table (without schema prefix)
        connection_string: ODBC connection string (if connection not provided)
        connection: Existing pyodbc connection (if connection_string not provided)
        if_exists: How to behave if the table already exists:
            - "fail": Raise an error (default)
            - "replace": Drop the table and recreate it
            - "append": Append data to the existing table
        schema: Database schema name (default: "dbo")
        batch_size: Number of rows to insert per batch (default: 10000)
        show_progress: Display progress bar during insertion (default: True)

    Returns:
        None

    Raises:
        ValueError: If neither connection_string nor connection is provided
        ValueError: If table exists and if_exists="fail"
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
        >>> write_database(df, "my_table", connection_string, if_exists="replace")
    """
    # Validate inputs
    if connection_string is None and connection is None:
        raise ValueError("Either connection_string or connection must be provided")

    if connection is None and connection_string is not None:
        should_close_connection = True
        connection = pyodbc.connect(connection_string)
    else:
        should_close_connection = False

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
                    f"Use if_exists='replace' or 'append' to modify this behavior."
                )
            elif if_exists == "replace":
                # Drop the existing table
                cursor.execute(f"DROP TABLE {full_table_name}")
                table_exists = False

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

        # Build INSERT statement
        columns = df.columns
        columns_str = ", ".join([f"[{col}]" for col in columns])
        placeholders = ", ".join(["?"] * len(columns))
        insert_sql = (
            f"INSERT INTO {full_table_name} ({columns_str}) VALUES ({placeholders})"
        )

        # Convert DataFrame to list of tuples for insertion
        # Process in batches to manage memory
        num_batches = (num_rows + batch_size - 1) // batch_size

        # Create progress bar iterator
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
            # Using to_dicts() and converting to tuples is efficient
            rows = []
            for row_dict in batch_df.iter_rows(named=False):
                # Prepare each value in the row
                prepared_row = tuple(
                    prepare_value_for_insert(val, dtype)
                    for val, dtype in zip(row_dict, df.dtypes)
                )
                rows.append(prepared_row)

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
