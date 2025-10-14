import polars as pl



@pl.api.register_expr_namespace("num_ext") 
class NumericExtensionNamespace: 
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    # Roman numeral mappings
    _roman_map = [
        ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),
        ("C", 100), ("XC", 90), ("L", 50), ("XL", 40),
        ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1)
    ]

    def to_roman(self) -> pl.Expr:
        """
        Convert an integer to Roman numerals.

        Examples
        --------
        .. code-block:: python
        
            import polars as pl
            import polars_extensions as plx
            df = pl.DataFrame({"numbers": [1, 2, 309, 4, 5]})
            result = df.with_columns(
                pl.col('numbers').num_ext.to_roman().alias("Roman")
            )

            result

        .. code-block:: text 

            shape: (5, 2)
            ┌─────────┬───────┐
            │ numbers ┆ Roman │
            │ ---     ┆ ---   │
            │ i64     ┆ str   │
            ╞═════════╪═══════╡
            │ 1       ┆ I     │
            │ 2       ┆ II    │
            │ 309     ┆ CCCIX │
            │ 4       ┆ IV    │
            │ 5       ┆ V     │
            └─────────┴───────┘
        """
        def convert_to_roman(value: int) -> str:
            if not (0 < value < 4000):
                raise ValueError("Number out of range (1-3999)")

            result = []
            for roman, num in self._roman_map:
                while value >= num:
                    result.append(roman)
                    value -= num
            return ''.join(result)

        # Use map_elements for element-wise operation
        return self._expr.map_elements(convert_to_roman,return_dtype=pl.String)

    def from_roman(self) -> pl.Expr:
        """
        Convert Roman numerals to integers.

        Examples
        --------
        .. code-block:: python
        
            import polars_extensions as plx

            df = pl.DataFrame({"Roman": ['I', 'II', 'III', 'CCCIX', 'V']})
            result = df.with_columns(
                pl.col('Roman').num_ext.from_roman().alias("Decoded")
            )

            result

        .. code-block:: text 

            shape: (5, 2)
            ┌───────┬─────────┐
            │ Roman ┆ Decoded │
            │ ---   ┆ ---     │
            │ str   ┆ i64     │
            ╞═══════╪═════════╡
            │ I     ┆ 1       │
            │ II    ┆ 2       │
            │ III   ┆ 3       │
            │ CCCIX ┆ 309     │
            │ V     ┆ 5       │
            └───────┴─────────┘
        """
        roman_to_value = {roman: value for roman, value in self._roman_map}

        def convert_from_roman(roman: str) -> int:
            i = 0
            total = 0
            while i < len(roman):
                # Check for two-character numeral first
                if i + 1 < len(roman) and roman[i:i+2] in roman_to_value:
                    total += roman_to_value[roman[i:i+2]]
                    i += 2
                else:
                    total += roman_to_value[roman[i]]
                    i += 1
            return total

        # Use map_elements for element-wise operation
        return self._expr.map_elements(convert_from_roman,return_dtype=pl.Int64)
    
    def word_to_number(self) -> pl.Expr:
        """Convert Natural Language to Numbers

        Examples
        --------

        .. code-block:: python

            import polars as pl
            import polars_extensions as plx

            df = pl.DataFrame({"numbers": ['6', 'two', 'three hundred and nine', '5', '4']})
            df.with_columns(
                pl.col('numbers').num_ext.word_to_number().alias("Actual Numbers")
            )

        .. code-block:: text 

            shape: (5, 2)
            ┌────────────────────────┬────────────────┐
            │ numbers                ┆ Actual Numbers │
            │ ---                    ┆ ---            │
            │ str                    ┆ i64            │
            ╞════════════════════════╪════════════════╡
            │ 6                      ┆ 6              │
            │ two                    ┆ 2              │
            │ three hundred and nine ┆ 309            │
            │ 5                      ┆ 5              │
            │ 4                      ┆ 4              │
            └────────────────────────┴────────────────┘
            
        
        """
        from word2number import w2n
        
        return self._expr.map_elements(lambda x: w2n.word_to_num(x) if isinstance(x, str) else x, return_dtype=pl.Int64)
