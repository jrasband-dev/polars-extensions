import polars as pl
import urllib.parse as urlparse

@pl.api.register_expr_namespace("url_ext")
class UrlExtensionNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def scheme(self) -> pl.Expr:
        """Extract URL scheme (http, https, ftp, etc.)."""
        return self._expr.map_elements(
            lambda x: urlparse.urlparse(x).scheme if x else None,
            return_dtype=pl.Utf8
        )

    def host(self) -> pl.Expr:
        """Extract host (domain + optional port)."""
        return self._expr.map_elements(
            lambda x: urlparse.urlparse(x).netloc if x else None,
            return_dtype=pl.Utf8
        )

    def domain(self) -> pl.Expr:
        """Extract just the domain name (without port or subdomain)."""
        def get_domain(u):
            if not u:
                return None
            netloc = urlparse.urlparse(u).netloc
            parts = netloc.split('.')
            return '.'.join(parts[-2:]) if len(parts) >= 2 else netloc
        return self._expr.map_elements(get_domain, return_dtype=pl.Utf8)

    def path(self) -> pl.Expr:
        """Extract the path (/docs/, /path/to/page, etc.)."""
        return self._expr.map_elements(
            lambda x: urlparse.urlparse(x).path if x else None,
            return_dtype=pl.Utf8
        )

    def query(self) -> pl.Expr:
        """
        Extract the full query string.

        Examples
        --------
        .. code-block:: python

            import polars as pl
            import polars_extensions as plx

            df = pl.DataFrame({
                "link": [
                    "https://pypi.org/search/?q=polars-extensions",
                    "https://pypi.org/search/?q=polars",
                    "https://pypi.org/search/?q=pyodbc"
                ]
            })
            df.with_columns(pl.col('link').url_ext.query().alias('q'))

        .. code-block:: text

            shape: (3, 2)
            ┌─────────────────────────────────┬─────────────────────┐
            │ link                            ┆ q                   │
            │ ---                             ┆ ---                 │
            │ str                             ┆ str                 │
            ╞═════════════════════════════════╪═════════════════════╡
            │ https://pypi.org/search/?q=pol… ┆ q=polars-extensions │
            │ https://pypi.org/search/?q=pol… ┆ q=polars            │
            │ https://pypi.org/search/?q=pyo… ┆ q=pyodbc            │
            └─────────────────────────────────┴─────────────────────┘
        
        """
        return self._expr.map_elements(
            lambda x: urlparse.urlparse(x).query if x else None,
            return_dtype=pl.Utf8
        )

    def query_param(self, key: str) -> pl.Expr:
        """
        
        Extract the value of a specific query parameter.

        Examples
        --------
        .. code-block:: python

            import polars as pl
            import polars_extensions as plx

            df = pl.DataFrame({
            "link": [
            "https://pypi.org/search/?q=polars-extensions",
            "https://pypi.org/search/?q=polars",
            "https://pypi.org/search/?q=pyodbc"
            ]
            })
            df.with_columns(pl.col('link').url_ext.query_param('q').alias('q'))
        

        .. code-block:: text

            shape: (3, 2)
            ┌─────────────────────────────────┬───────────────────┐
            │ link                            ┆ q                 │
            │ ---                             ┆ ---               │
            │ str                             ┆ str               │
            ╞═════════════════════════════════╪═══════════════════╡
            │ https://pypi.org/search/?q=pol… ┆ polars-extensions │
            │ https://pypi.org/search/?q=pol… ┆ polars            │
            │ https://pypi.org/search/?q=pyo… ┆ pyodbc            │
            └─────────────────────────────────┴───────────────────┘
        """
        def extract_param(u):
            if not u:
                return None
            params = urlparse.parse_qs(urlparse.urlparse(u).query)
            val = params.get(key)
            return val[0] if val else None
        return self._expr.map_elements(extract_param, return_dtype=pl.Utf8)

    def fragment(self) -> pl.Expr:
        """Extract fragment (#section)."""
        return self._expr.map_elements(
            lambda x: urlparse.urlparse(x).fragment if x else None,
            return_dtype=pl.Utf8
        )
    
