import polars as pl
import re


@pl.api.register_dataframe_namespace("name_ext")
class NameExtensionNameSpace:
    "Functions that extend the Name capabilities of polars DataFrames"

    def __init__(self, df: pl.DataFrame):
        self._df = df

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
        def _to_pascal_case(name: str) -> str:
            return "".join(
                word.capitalize() for word in re.sub(r"[_\s]+", " ", name).split()
            )

        columns = self._df.columns
        new_columns = {col: _to_pascal_case(col) for col in columns}
        return self._df.rename(new_columns)

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
    
        def _to_snake_case(name: str) -> str:
            return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

        new_columns = {col: _to_snake_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)

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
        def _to_camel_case(name: str) -> str:
            words = re.sub(r"[_\s]+", " ", name).split()
            return words[0].lower() + "".join(word.capitalize() for word in words[1:])

        new_columns = {col: _to_camel_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)

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
        def _to_pascal_snake_case(name: str) -> str:
            words = re.sub(r"[_\s]+", " ", name).split()
            return "_".join(word.capitalize() for word in words)

        new_columns = {col: _to_pascal_snake_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)

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
        def _to_kebeb_case(name: str) -> str:
            return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower().replace("_", "-")

        new_columns = {col: _to_kebeb_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)

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
        def _to_upper_snake_case(name: str) -> str:
            return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper().replace("-", "_")

        new_columns = {col: _to_upper_snake_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)

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
        def _to_train_case(name: str) -> str:
            return "-".join(
                word.capitalize() for word in re.sub(r"[_\s]+", " ", name).split()
            )

        new_columns = {col: _to_train_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)
