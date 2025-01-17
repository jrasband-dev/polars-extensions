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
        
        Parameters:
        col_a (str): The name of the first column to compare.
        col_b (str): The name of the second column to compare.
        
        Returns:
        pl.DataFrame: A DataFrame with the similarity scores as a new column.
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