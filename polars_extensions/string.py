import polars as pl

@pl.api.register_dataframe_namespace("str_ext")
class StringExtensionNamespace:
    """String Extensions for the Polars Library"""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def f1_string_similarity(self, col_a: str, col_b: str) -> pl.DataFrame:
        """
        Calculates a similarity score between two columns of strings based on common characters,
        accounting for repeated characters.
        
        Parameters
        ----------
        col_a (str): The name of the first column to compare.
        col_b (str): The name of the second column to compare.
        
        Returns
        -------
        DataFrame

        Examples
        --------

        .. code-block:: python

            import polars_extensions as plx
            import polars as pl 

            data = pl.read_csv('datasets/string_sim.csv')
            data.str_ext.f1_string_similarity('a','c')

        .. code-block:: text

            shape: (13, 3)
            ┌──────────────────────────┬───────────────────────────┬──────────┐
            │ a                        ┆ c                         ┆ f1_score │
            │ ---                      ┆ ---                       ┆ ---      │
            │ str                      ┆ str                       ┆ f64      │
            ╞══════════════════════════╪═══════════════════════════╪══════════╡
            │ apple                    ┆ appl                      ┆ 0.888889 │
            │ banana                   ┆ BANANA                    ┆ 1.0      │
            │ cherry                   ┆ cherr                     ┆ 0.909091 │
            │ date                     ┆ etad                      ┆ 1.0      │
            │ elderberry               ┆ elderberrys               ┆ 0.952381 │
            │ …                        ┆ …                         ┆ …        │
            │ kiwi                     ┆ KIW                       ┆ 0.857143 │
            │ lemon                    ┆ lemons                    ┆ 0.909091 │
            │ mangoes are Tangy        ┆ mango are Tangy           ┆ 0.9375   │
            │ it was the best of times ┆ it was the worst of times ┆ 0.897959 │
            │ of times it was the best ┆ it was the worst of times ┆ 0.897959 │
            └──────────────────────────┴───────────────────────────┴──────────┘
        

        """

        def similarity(row_str_a: str, row_str_b: str) -> float:
            # Normalize both strings (case-insensitive comparison)
            row_str_a = row_str_a.lower()
            row_str_b = row_str_b.lower()

            # If strings are identical, return a score of 1.0
            if row_str_a == row_str_b:
                return 1.0

            list1 = list(row_str_a)
            list2 = list(row_str_b)

            list2_copy = list2[:]
            intersection = []

            # Account for repeated characters by checking all occurrences
            for char in list1:
                if char in list2_copy:
                    intersection.append(char)
                    list2_copy.remove(char)
            
            common_chars = len(intersection)
            total_chars = len(list1) + len(list2)
            return (2 * common_chars) / total_chars if total_chars > 0 else 0.0

        # Apply the similarity function row-by-row
        similarity_scores = [
            similarity(row_a, row_b) for row_a, row_b in zip(self._df[col_a], self._df[col_b])
        ]

        # Add the similarity scores as a new column to the DataFrame
        self._df = self._df.with_columns(
            pl.Series("f1_score", similarity_scores)
        )

        return self._df