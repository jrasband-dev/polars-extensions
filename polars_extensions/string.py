import polars as pl


@pl.api.register_expr_namespace("str_ext")
class StringExtensionNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    # Helper method for Levenshtein Distance
    @staticmethod
    def _levenshtein_distance(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j  # Insert all remaining chars of b
                elif j == 0:
                    dp[i][j] = i  # Remove all remaining chars of a
                elif a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No cost
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],  # Deletion
                        dp[i][j - 1],  # Insertion
                        dp[i - 1][j - 1],  # Substitution
                    )
        return dp[m][n]

    # Levenshtein Distance Expression
    def levenshtein_distance(self, other: pl.Expr) -> pl.Expr:
        return pl.struct([self._expr, other]).map_elements(
            lambda x: StringExtensionNamespace._levenshtein_distance(x[0], x[1])
        )

    # Helper method for Jaccard Similarity
    @staticmethod
    def _jaccard_similarity(a: str, b: str, n: int = 2) -> float:
        def ngrams(string, n):
            return set(string[i : i + n] for i in range(len(string) - n + 1))

        a_ngrams = ngrams(a, n)
        b_ngrams = ngrams(b, n)
        intersection = len(a_ngrams & b_ngrams)
        union = len(a_ngrams | b_ngrams)
        return intersection / union if union != 0 else 0.0

    # Jaccard Similarity Expression
    def jaccard_similarity(self, other: pl.Expr, n: int = 2) -> pl.Expr:
        return pl.struct([self._expr, other]).map_elements(
            lambda x: StringExtensionNamespace._jaccard_similarity(x[0], x[1], n)
        )
