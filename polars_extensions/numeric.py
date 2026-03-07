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


def _map_with_rust(expr: pl.Expr, rust_fn_name: str, return_dtype: pl.DataType) -> pl.Expr:
    if _name_rust is None:
        raise ImportError(
            "Rust extension module 'polars_extensions._name_rust' is required for num_ext. "
            "Install or reinstall `polars-extensions` from a wheel for your platform. "
            "For local development, run `python -m maturin develop -m Cargo.toml`."
        )

    if not hasattr(_name_rust, rust_fn_name):
        raise AttributeError(
            f"Rust converter '{rust_fn_name}' is not available in polars_extensions._name_rust"
        )

    rust_fn = getattr(_name_rust, rust_fn_name)
    return expr.map_elements(rust_fn, return_dtype=return_dtype)


def to_roman(expr: pl.Expr) -> pl.Expr:
    return _map_with_rust(expr, "to_roman_scalar", pl.String)


def from_roman(expr: pl.Expr) -> pl.Expr:
    return _map_with_rust(expr, "from_roman_scalar", pl.Int64)


def word_to_number(expr: pl.Expr) -> pl.Expr:
    return _map_with_rust(expr, "word_to_number_scalar", pl.Int64)


@pl.api.register_expr_namespace("num_ext")
class NumericExtensionNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

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

        return to_roman(self._expr)

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
        return from_roman(self._expr)

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
        return word_to_number(self._expr)
