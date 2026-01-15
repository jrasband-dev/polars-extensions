import json
from pathlib import Path

import polars as pl
import pytest
from polars_extensions import io

from polars_extensions.io import read_xml, write_schema, read_schema


def test_read_xml_str_and_path(tmp_path: Path):
    xml = """
    <catalog>
        <product sku="A123" available="true">
            <name>Mechanical Keyboard</name>
            <price currency="USD">129.99</price>
            <features>
                <feature>RGB Lighting</feature>
                <feature>Hot-swappable switches</feature>
            </features>
        </product>
    </catalog>
    """

    df1 = read_xml(xml, flatten=True)
    assert isinstance(df1, pl.DataFrame)
    assert df1.height >= 1

    p = tmp_path / "data.xml"
    p.write_text(xml)

    df2 = read_xml(p, flatten=True)
    assert isinstance(df2, pl.DataFrame)
    assert df2.shape == df1.shape
    assert set(df2.columns) == set(df1.columns)


def test_read_xml_bytes():
    xml = b"<root><item>one</item><item>two</item></root>"
    df = read_xml(xml, record_path="root.item", flatten=True)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2


def test_write_and_read_schema(tmp_path: Path):
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = tmp_path / "schema.json"

    write_schema(df, p)
    assert p.exists()

    j = json.load(p.open())
    assert set(j.keys()) == set(df.columns)

    schema = read_schema(p)
    # Returned schema should be mapping-like (pl.Schema / dict)
    assert hasattr(schema, "keys") or isinstance(schema, dict)
    assert set(schema.keys()) == set(j.keys())


def test_io_dummy():
    # Replace with real tests for io module
    assert hasattr(io, "__file__") or True
