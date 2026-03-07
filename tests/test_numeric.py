import polars as pl
import pytest
import polars_extensions
import polars_extensions.numeric as numeric_module


if numeric_module._name_rust is None:
    pytest.skip("Rust extension module _name_rust is required for numeric tests", allow_module_level=True)


def test_to_roman() -> None:
    df = pl.DataFrame({"numbers": [1, 2, 309, 4, 5]})
    result = df.with_columns(pl.col("numbers").num_ext.to_roman().alias("roman"))
    assert result["roman"].to_list() == ["I", "II", "CCCIX", "IV", "V"]


def test_from_roman() -> None:
    df = pl.DataFrame({"roman": ["I", "II", "III", "CCCIX", "V"]})
    result = df.with_columns(pl.col("roman").num_ext.from_roman().alias("decoded"))
    assert result["decoded"].to_list() == [1, 2, 3, 309, 5]


def test_to_roman_out_of_range_raises() -> None:
    df = pl.DataFrame({"numbers": [0]})
    with pytest.raises(Exception):
        df.with_columns(pl.col("numbers").num_ext.to_roman().alias("roman"))


def test_word_to_number() -> None:
    df = pl.DataFrame(
        {
            "numbers": [
                "6",
                "two",
                "three hundred and nine",
                "twenty-one",
            ]
        }
    )
    result = df.with_columns(pl.col("numbers").num_ext.word_to_number().alias("actual"))
    assert result["actual"].to_list() == [6, 2, 309, 21]


def test_word_to_number_invalid_raises() -> None:
    df = pl.DataFrame({"numbers": ["not_a_number_word"]})
    with pytest.raises(Exception):
        df.with_columns(pl.col("numbers").num_ext.word_to_number().alias("actual"))
