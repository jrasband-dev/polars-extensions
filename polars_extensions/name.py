import polars as pl
import importlib
import importlib.util
import os
import sys
from glob import glob


def _load_rust_extension():
    try:
        from . import _name_rust as module

        return module
    except ImportError:
        pass

    for module_name in ("polars_extensions._name_rust", "_name_rust"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue

    for path_entry in sys.path:
        if "site-packages" not in path_entry.lower():
            continue

        candidates = glob(os.path.join(path_entry, "polars_extensions", "_name_rust*.pyd"))
        candidates += glob(os.path.join(path_entry, "polars_extensions", "_name_rust*.so"))
        candidates += glob(os.path.join(path_entry, "polars_extensions", "_name_rust*.dylib"))

        for candidate in candidates:
            spec = importlib.util.spec_from_file_location(
                "polars_extensions._name_rust",
                candidate,
            )
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    return None


_name_rust = _load_rust_extension()


@pl.api.register_dataframe_namespace("name_ext")
class NameExtensionNameSpace:
    "Functions that extend the Name capabilities of polars DataFrames"

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def _rename_columns_with(
        self,
        rust_converter_name: str,
    ) -> pl.DataFrame:
        if _name_rust is None:
            raise ImportError(
                "Rust extension module 'polars_extensions._name_rust' is required for name_ext. "
                "Install or reinstall `polars-extensions` from a wheel for your platform. "
                "For local development, run `python -m maturin develop -m Cargo.toml`."
            )

        columns = self._df.columns
        if not hasattr(_name_rust, rust_converter_name):
            raise AttributeError(
                f"Rust converter '{rust_converter_name}' is not available in polars_extensions._name_rust"
            )

        converter = getattr(_name_rust, rust_converter_name)
        converted = converter(columns)

        return self._df.rename(dict(zip(columns, converted)))

    def to_pascal_case(self) -> pl.DataFrame:
        """Converts column names to PascalCase

        Examples
        --------
        .. code-block:: python

            import polars as pl
            from polars_extensions import *
            data = pl.read_csv('datasets/employees.csv')
            data.name_ext.to_pascal_case()

        .. code-block:: text

            shape: (5, 8)
            ┌────────────┬───────────┬──────────┬─────────────┬─────────────┬────────────┬────────────┬────────┐
            │ EmployeeId ┆ FirstName ┆ LastName ┆ Email       ┆ JobTitle    ┆ DateOfBirt ┆ DateOfHire ┆ Salary │
            │ ---        ┆ ---       ┆ ---      ┆ ---         ┆ ---         ┆ h          ┆ ---        ┆ ---    │
            │ i64        ┆ str       ┆ str      ┆ str         ┆ str         ┆ ---        ┆ str        ┆ i64    │
            │            ┆           ┆          ┆             ┆             ┆ str        ┆            ┆        │
            ╞════════════╪═══════════╪══════════╪═════════════╪═════════════╪════════════╪════════════╪════════╡
            │ 1          ┆ john      ┆ doe      ┆ john.doe@ex ┆ software_en ┆ 1990-05-12 ┆ 2015-08-01 ┆ 85000  │
            │            ┆           ┆          ┆ ample.com   ┆ gineer      ┆            ┆            ┆        │
            │ 2          ┆ jane      ┆ smith    ┆ jane.smith@ ┆ data_scient ┆ 1988-11-23 ┆ 2017-03-15 ┆ 95000  │
            │            ┆           ┆          ┆ example.com ┆ ist         ┆            ┆            ┆        │
            │ 3          ┆ bob       ┆ johnson  ┆ bob.johnson ┆ product_man ┆ 1985-07-19 ┆ 2012-10-10 ┆ 105000 │
            │            ┆           ┆          ┆ @example.co ┆ ager        ┆            ┆            ┆        │
            │            ┆           ┆          ┆ m           ┆             ┆            ┆            ┆        │
            │ 4          ┆ alice     ┆ davis    ┆ alice.davis ┆ ux_designer ┆ 1992-04-06 ┆ 2020-01-21 ┆ 78000  │
            │            ┆           ┆          ┆ @example.co ┆             ┆            ┆            ┆        │
            │            ┆           ┆          ┆ m           ┆             ┆            ┆            ┆        │
            │ 5          ┆ charlie   ┆ brown    ┆ charlie.bro ┆ qa_engineer ┆ 1993-09-14 ┆ 2019-07-08 ┆ 72000  │
            │            ┆           ┆          ┆ wn@example. ┆             ┆            ┆            ┆        │
            │            ┆           ┆          ┆ com         ┆             ┆            ┆            ┆        │
            └────────────┴───────────┴──────────┴─────────────┴─────────────┴────────────┴────────────┴────────┘

        """

        return self._rename_columns_with("to_pascal_case_columns")

    def to_snake_case(self) -> pl.DataFrame:
        """Converts column names to snake_case

        Examples
        --------
        .. code-block:: python

            import polars as pl
            from polars_extensions import *
            data = pl.read_csv('datasets/employees.csv')
            data.name_ext.to_snake_case()

        .. code-block:: text

            shape: (5, 8)
            ┌────────────┬────────────┬───────────┬────────────┬────────────┬────────────┬────────────┬────────┐
            │ employee_i ┆ first_name ┆ last_name ┆ email      ┆ job_title  ┆ date_of_bi ┆ date_of_hi ┆ salary │
            │ d          ┆ ---        ┆ ---       ┆ ---        ┆ ---        ┆ rth        ┆ re         ┆ ---    │
            │ ---        ┆ str        ┆ str       ┆ str        ┆ str        ┆ ---        ┆ ---        ┆ i64    │
            │ i64        ┆            ┆           ┆            ┆            ┆ str        ┆ str        ┆        │
            ╞════════════╪════════════╪═══════════╪════════════╪════════════╪════════════╪════════════╪════════╡
            │ 1          ┆ john       ┆ doe       ┆ john.doe@e ┆ software_e ┆ 1990-05-12 ┆ 2015-08-01 ┆ 85000  │
            │            ┆            ┆           ┆ xample.com ┆ ngineer    ┆            ┆            ┆        │
            │ 2          ┆ jane       ┆ smith     ┆ jane.smith ┆ data_scien ┆ 1988-11-23 ┆ 2017-03-15 ┆ 95000  │
            │            ┆            ┆           ┆ @example.c ┆ tist       ┆            ┆            ┆        │
            │            ┆            ┆           ┆ om         ┆            ┆            ┆            ┆        │
            │ 3          ┆ bob        ┆ johnson   ┆ bob.johnso ┆ product_ma ┆ 1985-07-19 ┆ 2012-10-10 ┆ 105000 │
            │            ┆            ┆           ┆ n@example. ┆ nager      ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 4          ┆ alice      ┆ davis     ┆ alice.davi ┆ ux_designe ┆ 1992-04-06 ┆ 2020-01-21 ┆ 78000  │
            │            ┆            ┆           ┆ s@example. ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 5          ┆ charlie    ┆ brown     ┆ charlie.br ┆ qa_enginee ┆ 1993-09-14 ┆ 2019-07-08 ┆ 72000  │
            │            ┆            ┆           ┆ own@exampl ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ e.com      ┆            ┆            ┆            ┆        │
            └────────────┴────────────┴───────────┴────────────┴────────────┴────────────┴────────────┴────────┘
        """

        return self._rename_columns_with("to_snake_case_columns")

    def to_camel_case(self) -> pl.DataFrame:
        """Converts column names to camelCase

        Examples
        --------

        .. code-block:: python

            import polars as pl
            from polars_extensions import *
            data = pl.read_csv('datasets/employees.csv')
            data.name_ext.to_camel_case()

        .. code-block:: text

            shape: (5, 8)
            ┌────────────┬───────────┬──────────┬─────────────┬─────────────┬────────────┬────────────┬────────┐
            │ employeeId ┆ firstName ┆ lastName ┆ email       ┆ jobTitle    ┆ dateOfBirt ┆ dateOfHire ┆ salary │
            │ ---        ┆ ---       ┆ ---      ┆ ---         ┆ ---         ┆ h          ┆ ---        ┆ ---    │
            │ i64        ┆ str       ┆ str      ┆ str         ┆ str         ┆ ---        ┆ str        ┆ i64    │
            │            ┆           ┆          ┆             ┆             ┆ str        ┆            ┆        │
            ╞════════════╪═══════════╪══════════╪═════════════╪═════════════╪════════════╪════════════╪════════╡
            │ 1          ┆ john      ┆ doe      ┆ john.doe@ex ┆ software_en ┆ 1990-05-12 ┆ 2015-08-01 ┆ 85000  │
            │            ┆           ┆          ┆ ample.com   ┆ gineer      ┆            ┆            ┆        │
            │ 2          ┆ jane      ┆ smith    ┆ jane.smith@ ┆ data_scient ┆ 1988-11-23 ┆ 2017-03-15 ┆ 95000  │
            │            ┆           ┆          ┆ example.com ┆ ist         ┆            ┆            ┆        │
            │ 3          ┆ bob       ┆ johnson  ┆ bob.johnson ┆ product_man ┆ 1985-07-19 ┆ 2012-10-10 ┆ 105000 │
            │            ┆           ┆          ┆ @example.co ┆ ager        ┆            ┆            ┆        │
            │            ┆           ┆          ┆ m           ┆             ┆            ┆            ┆        │
            │ 4          ┆ alice     ┆ davis    ┆ alice.davis ┆ ux_designer ┆ 1992-04-06 ┆ 2020-01-21 ┆ 78000  │
            │            ┆           ┆          ┆ @example.co ┆             ┆            ┆            ┆        │
            │            ┆           ┆          ┆ m           ┆             ┆            ┆            ┆        │
            │ 5          ┆ charlie   ┆ brown    ┆ charlie.bro ┆ qa_engineer ┆ 1993-09-14 ┆ 2019-07-08 ┆ 72000  │
            │            ┆           ┆          ┆ wn@example. ┆             ┆            ┆            ┆        │
            │            ┆           ┆          ┆ com         ┆             ┆            ┆            ┆        │
            └────────────┴───────────┴──────────┴─────────────┴─────────────┴────────────┴────────────┴────────┘
        """

        return self._rename_columns_with("to_camel_case_columns")

    def to_pascal_snake_case(self) -> pl.DataFrame:
        """Converts column names to Pascal_Snake_Case

        Examples
        --------
        .. code-block:: python

            import polars as pl
            from polars_extensions import *
            data = pl.read_csv('datasets/employees.csv')
            data.name_ext.to_pascal_snake_case()

        .. code-block:: text

            shape: (5, 8)
            ┌────────────┬────────────┬───────────┬────────────┬────────────┬────────────┬────────────┬────────┐
            │ Employee_I ┆ First_Name ┆ Last_Name ┆ Email      ┆ Job_Title  ┆ Date_Of_Bi ┆ Date_Of_Hi ┆ Salary │
            │ d          ┆ ---        ┆ ---       ┆ ---        ┆ ---        ┆ rth        ┆ re         ┆ ---    │
            │ ---        ┆ str        ┆ str       ┆ str        ┆ str        ┆ ---        ┆ ---        ┆ i64    │
            │ i64        ┆            ┆           ┆            ┆            ┆ str        ┆ str        ┆        │
            ╞════════════╪════════════╪═══════════╪════════════╪════════════╪════════════╪════════════╪════════╡
            │ 1          ┆ john       ┆ doe       ┆ john.doe@e ┆ software_e ┆ 1990-05-12 ┆ 2015-08-01 ┆ 85000  │
            │            ┆            ┆           ┆ xample.com ┆ ngineer    ┆            ┆            ┆        │
            │ 2          ┆ jane       ┆ smith     ┆ jane.smith ┆ data_scien ┆ 1988-11-23 ┆ 2017-03-15 ┆ 95000  │
            │            ┆            ┆           ┆ @example.c ┆ tist       ┆            ┆            ┆        │
            │            ┆            ┆           ┆ om         ┆            ┆            ┆            ┆        │
            │ 3          ┆ bob        ┆ johnson   ┆ bob.johnso ┆ product_ma ┆ 1985-07-19 ┆ 2012-10-10 ┆ 105000 │
            │            ┆            ┆           ┆ n@example. ┆ nager      ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 4          ┆ alice      ┆ davis     ┆ alice.davi ┆ ux_designe ┆ 1992-04-06 ┆ 2020-01-21 ┆ 78000  │
            │            ┆            ┆           ┆ s@example. ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 5          ┆ charlie    ┆ brown     ┆ charlie.br ┆ qa_enginee ┆ 1993-09-14 ┆ 2019-07-08 ┆ 72000  │
            │            ┆            ┆           ┆ own@exampl ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ e.com      ┆            ┆            ┆            ┆        │
            └────────────┴────────────┴───────────┴────────────┴────────────┴────────────┴────────────┴────────┘
        """

        return self._rename_columns_with("to_pascal_snake_case_columns")

    def to_kebeb_case(self) -> pl.DataFrame:
        """Converts column names to kebab-case

        Examples
        --------
        .. code-block:: python

            import polars as pl
            from polars_extensions import *
            data = pl.read_csv('datasets/employees.csv')
            data.name_ext.to_kebeb_case()

        .. code-block:: text

            shape: (5, 8)
            ┌────────────┬────────────┬───────────┬────────────┬────────────┬────────────┬────────────┬────────┐
            │ employee-i ┆ first-name ┆ last-name ┆ email      ┆ job-title  ┆ date-of-bi ┆ date-of-hi ┆ salary │
            │ d          ┆ ---        ┆ ---       ┆ ---        ┆ ---        ┆ rth        ┆ re         ┆ ---    │
            │ ---        ┆ str        ┆ str       ┆ str        ┆ str        ┆ ---        ┆ ---        ┆ i64    │
            │ i64        ┆            ┆           ┆            ┆            ┆ str        ┆ str        ┆        │
            ╞════════════╪════════════╪═══════════╪════════════╪════════════╪════════════╪════════════╪════════╡
            │ 1          ┆ john       ┆ doe       ┆ john.doe@e ┆ software_e ┆ 1990-05-12 ┆ 2015-08-01 ┆ 85000  │
            │            ┆            ┆           ┆ xample.com ┆ ngineer    ┆            ┆            ┆        │
            │ 2          ┆ jane       ┆ smith     ┆ jane.smith ┆ data_scien ┆ 1988-11-23 ┆ 2017-03-15 ┆ 95000  │
            │            ┆            ┆           ┆ @example.c ┆ tist       ┆            ┆            ┆        │
            │            ┆            ┆           ┆ om         ┆            ┆            ┆            ┆        │
            │ 3          ┆ bob        ┆ johnson   ┆ bob.johnso ┆ product_ma ┆ 1985-07-19 ┆ 2012-10-10 ┆ 105000 │
            │            ┆            ┆           ┆ n@example. ┆ nager      ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 4          ┆ alice      ┆ davis     ┆ alice.davi ┆ ux_designe ┆ 1992-04-06 ┆ 2020-01-21 ┆ 78000  │
            │            ┆            ┆           ┆ s@example. ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 5          ┆ charlie    ┆ brown     ┆ charlie.br ┆ qa_enginee ┆ 1993-09-14 ┆ 2019-07-08 ┆ 72000  │
            │            ┆            ┆           ┆ own@exampl ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ e.com      ┆            ┆            ┆            ┆        │
            └────────────┴────────────┴───────────┴────────────┴────────────┴────────────┴────────────┴────────┘
        """

        return self._rename_columns_with("to_kebab_case_columns")

    def to_upper_snake_case(self) -> pl.DataFrame:
        """Converts column names to UPPER_SNAKE_CASE

        Examples
        --------
        .. code-block:: python

            import polars as pl
            from polars_extensions import *
            data = pl.read_csv('datasets/employees.csv')
            data.name_ext.to_kebeb_case()

        .. code-block:: text

            shape: (5, 8)
            ┌────────────┬────────────┬───────────┬────────────┬────────────┬────────────┬────────────┬────────┐
            │ EMPLOYEE_I ┆ FIRST_NAME ┆ LAST_NAME ┆ EMAIL      ┆ JOB_TITLE  ┆ DATE_OF_BI ┆ DATE_OF_HI ┆ SALARY │
            │ D          ┆ ---        ┆ ---       ┆ ---        ┆ ---        ┆ RTH        ┆ RE         ┆ ---    │
            │ ---        ┆ str        ┆ str       ┆ str        ┆ str        ┆ ---        ┆ ---        ┆ i64    │
            │ i64        ┆            ┆           ┆            ┆            ┆ str        ┆ str        ┆        │
            ╞════════════╪════════════╪═══════════╪════════════╪════════════╪════════════╪════════════╪════════╡
            │ 1          ┆ john       ┆ doe       ┆ john.doe@e ┆ software_e ┆ 1990-05-12 ┆ 2015-08-01 ┆ 85000  │
            │            ┆            ┆           ┆ xample.com ┆ ngineer    ┆            ┆            ┆        │
            │ 2          ┆ jane       ┆ smith     ┆ jane.smith ┆ data_scien ┆ 1988-11-23 ┆ 2017-03-15 ┆ 95000  │
            │            ┆            ┆           ┆ @example.c ┆ tist       ┆            ┆            ┆        │
            │            ┆            ┆           ┆ om         ┆            ┆            ┆            ┆        │
            │ 3          ┆ bob        ┆ johnson   ┆ bob.johnso ┆ product_ma ┆ 1985-07-19 ┆ 2012-10-10 ┆ 105000 │
            │            ┆            ┆           ┆ n@example. ┆ nager      ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 4          ┆ alice      ┆ davis     ┆ alice.davi ┆ ux_designe ┆ 1992-04-06 ┆ 2020-01-21 ┆ 78000  │
            │            ┆            ┆           ┆ s@example. ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 5          ┆ charlie    ┆ brown     ┆ charlie.br ┆ qa_enginee ┆ 1993-09-14 ┆ 2019-07-08 ┆ 72000  │
            │            ┆            ┆           ┆ own@exampl ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ e.com      ┆            ┆            ┆            ┆        │
            └────────────┴────────────┴───────────┴────────────┴────────────┴────────────┴────────────┴────────┘


        """

        return self._rename_columns_with("to_upper_snake_case_columns")

    def to_train_case(self) -> pl.DataFrame:
        """Converts column names to Train-Case

        Examples
        --------
        .. code-block:: python

            import polars as pl
            from polars_extensions import *
            data = pl.read_csv('datasets/employees.csv')
            data.name_ext.to_train_case()
        .. code-block:: text

            shape: (5, 8)
            ┌────────────┬────────────┬───────────┬────────────┬────────────┬────────────┬────────────┬────────┐
            │ Employee-I ┆ First-Name ┆ Last-Name ┆ Email      ┆ Job-Title  ┆ Date-Of-Bi ┆ Date-Of-Hi ┆ Salary │
            │ d          ┆ ---        ┆ ---       ┆ ---        ┆ ---        ┆ rth        ┆ re         ┆ ---    │
            │ ---        ┆ str        ┆ str       ┆ str        ┆ str        ┆ ---        ┆ ---        ┆ i64    │
            │ i64        ┆            ┆           ┆            ┆            ┆ str        ┆ str        ┆        │
            ╞════════════╪════════════╪═══════════╪════════════╪════════════╪════════════╪════════════╪════════╡
            │ 1          ┆ john       ┆ doe       ┆ john.doe@e ┆ software_e ┆ 1990-05-12 ┆ 2015-08-01 ┆ 85000  │
            │            ┆            ┆           ┆ xample.com ┆ ngineer    ┆            ┆            ┆        │
            │ 2          ┆ jane       ┆ smith     ┆ jane.smith ┆ data_scien ┆ 1988-11-23 ┆ 2017-03-15 ┆ 95000  │
            │            ┆            ┆           ┆ @example.c ┆ tist       ┆            ┆            ┆        │
            │            ┆            ┆           ┆ om         ┆            ┆            ┆            ┆        │
            │ 3          ┆ bob        ┆ johnson   ┆ bob.johnso ┆ product_ma ┆ 1985-07-19 ┆ 2012-10-10 ┆ 105000 │
            │            ┆            ┆           ┆ n@example. ┆ nager      ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 4          ┆ alice      ┆ davis     ┆ alice.davi ┆ ux_designe ┆ 1992-04-06 ┆ 2020-01-21 ┆ 78000  │
            │            ┆            ┆           ┆ s@example. ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ com        ┆            ┆            ┆            ┆        │
            │ 5          ┆ charlie    ┆ brown     ┆ charlie.br ┆ qa_enginee ┆ 1993-09-14 ┆ 2019-07-08 ┆ 72000  │
            │            ┆            ┆           ┆ own@exampl ┆ r          ┆            ┆            ┆        │
            │            ┆            ┆           ┆ e.com      ┆            ┆            ┆            ┆        │
            └────────────┴────────────┴───────────┴────────────┴────────────┴────────────┴────────────┴────────┘
        """

        return self._rename_columns_with("to_train_case_columns")
