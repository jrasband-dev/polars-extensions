# Polars Extensions

[![PyPI version](https://img.shields.io/pypi/v/polars-extensions.svg)](https://pypi.org/project/polars-extensions/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Documentation Status](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://jrasband-dev.github.io/polars-extensions/)

  
For full details and usage, see the [official documentation](https://jrasband-dev.github.io/polars-extensions/).

---

## Getting Started

```bash
pip install polars-extensions
```

---

## Why use Polars Extensions?

Polars Extensions fills in widely used convenience functions and transformations that many data workflows rely on — without needing custom ad-hoc code. Whether you need to:

- Read or write schema definitions to manage data structure programmatically  
- Convert between number formats (e.g. numbers ↔ Roman numerals, parse textual numbers)  
- Uniformly rename columns across naming conventions (snake_case, camelCase, kebab-case, PascalCase, etc.)  
- Compute approximate string-similarity (F1 similarity) for fuzzy matching or cleanup  
- Work with simple geospatial transformations (e.g. WKT/WKB ↔ coordinate formats)  

Polars Extensions provides a clean, expressive, and consistent API (using Polars expression namespaces) so you stay in the “Polars way” of data manipulation. 

