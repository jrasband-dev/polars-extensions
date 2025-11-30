import polars as pl

@pl.api.register_dataframe_namespace("ta_ext")
class TechnicalAnalysisNamespace:
    """Technical Analysis Extensions for the Polars Library"""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    # Example indicator --- delta
    def delta(self, col: str, periods: int = 1) -> pl.DataFrame:
        """
        Computes the difference between the current value and the value N periods ago.

        Parameters
        ----------
        col : str
            The column to compute delta from.
        periods : int, default 1
            Number of periods to shift.

        Returns
        -------
        DataFrame
        """
        return self._df.with_columns(
            (pl.col(col) - pl.col(col).shift(periods)).alias(f"{col}_delta_{periods}")
        )

    # Example indicator --- logarithmic return
    def log_return(self, col: str) -> pl.DataFrame:
        """
        Computes log returns using ln(close / close.shift(1)).
        """
        return self._df.with_columns(
            (pl.col(col) / pl.col(col).shift(1))
            .log()
            .alias(f"{col}_log_return")
        )

    # Example indicator --- SMA
    def sma(self, col: str, window: int) -> pl.DataFrame:
        """
        Simple Moving Average (SMA)
        """
        return self._df.with_columns(
            pl.col(col)
            .rolling_mean(window)
            .alias(f"{col}_sma_{window}")
        )

    def rsi(self, col: str, window: int = 14) -> pl.DataFrame:
        """
        Computes the Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        where RS = avg_gain / avg_loss
        """
        diff = pl.col(col) - pl.col(col).shift(1)

        df = self._df.with_columns(
            diff.alias("_diff"),
            pl.when(diff > 0).then(diff).otherwise(0).alias("_gain"),
            pl.when(diff < 0).then(-diff).otherwise(0).alias("_loss"),
        )

        df = df.with_columns(
            pl.col("_gain").rolling_mean(window).alias("_avg_gain"),
            pl.col("_loss").rolling_mean(window).alias("_avg_loss"),
        )

        df = df.with_columns(
            (100 - 100 / (1 + (pl.col("_avg_gain") / pl.col("_avg_loss"))))
            .alias(f"{col}_rsi_{window}")
        ).drop(["_diff", "_gain", "_loss", "_avg_gain", "_avg_loss"])

        return df

    def bollinger(self, col: str, window: int = 20, num_std: float = 2.0) -> pl.DataFrame:
        """
        Computes Bollinger Bands: middle, upper, and lower.
        """
        mid = pl.col(col).rolling_mean(window)
        std = pl.col(col).rolling_std(window)

        return self._df.with_columns(
            mid.alias(f"{col}_bb_mid"),
            (mid + num_std * std).alias(f"{col}_bb_upper"),
            (mid - num_std * std).alias(f"{col}_bb_lower"),
        )
    
    def atr(self, high: str, low: str, close: str, window: int = 14) -> pl.DataFrame:
        """
        Computes the Average True Range (ATR).
        """
        df = self._df.with_columns(
            (pl.col(high) - pl.col(low)).alias("_hl"),
            (pl.col(high) - pl.col(close).shift(1)).abs().alias("_hc"),
            (pl.col(low) - pl.col(close).shift(1)).abs().alias("_lc"),
        )

        df = df.with_columns(
            pl.max_horizontal(["_hl", "_hc", "_lc"]).alias("_tr")
        )

        df = df.with_columns(
            pl.col("_tr").rolling_mean(window).alias("atr")
        ).drop(["_hl", "_hc", "_lc", "_tr"])

        return df
