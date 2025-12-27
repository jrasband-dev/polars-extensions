import json
from typing import Union, Optional, List
import xml.etree.ElementTree as ET
import polars as pl
from pathlib import Path


def read_xml(
    xml_input: Union[str, bytes, Path],
    record_path: Optional[str] = None,
    include_attributes: bool = True,
    flatten: bool = True,
    strict: bool = False,
) -> pl.DataFrame:
    """
    Reads and normalizes XML into a flat or semi-structured Polars DataFrame.

    Parameters
    ----------
    xml_input : str | bytes | Path
        XML string, bytes, or file path.
    record_path : str, optional
        Dot-separated path to record nodes (e.g., "channel.item").
        Can optionally include the root element (e.g., "catalog.product").
    include_attributes : bool
        Whether to include XML attributes in the output.
    flatten : bool
        Recursively explode lists and unnest structs.
    strict : bool
        True -> Polars raises on type mismatch.
        False -> wraps primitives in lists to avoid schema mismatch.

    Returns
    -------
    pl.DataFrame

    Examples
    --------

    .. code-block:: python

        import polars_extensions as plx
        xml_data = '''
        <catalog>
            <product sku="A123" available="true">
                <name>Mechanical Keyboard</name>
                <price currency="USD">129.99</price>
                <features>
                    <feature>RGB Lighting</feature>
                    <feature>Hot-swappable switches</feature>
                    <feature>Aluminum frame</feature>
                </features>
            </product>

            <product sku="B456" available="false">
                <name>Noise Cancelling Headphones</name>
                <price currency="USD">299.00</price>
                <features>
                    <feature>ANC</feature>
                    <feature>Bluetooth 5.0</feature>
                </features>
            </product>
        </catalog>
        '''

        df = plx.read_xml(xml_data,flatten=True)
        df


    .. code-block:: text

        shape: (5, 6)
        ┌────────────────┬────────────────┬────────────────┬───────────────┬───────────────┬───────────────┐
        │ catalog.produc ┆ catalog.produc ┆ catalog.produc ┆ catalog.produ ┆ catalog.produ ┆ catalog.produ │
        │ t.product.sku  ┆ t.product.avai ┆ t.product.name ┆ ct.product.pr ┆ ct.product.pr ┆ ct.product.fe │
        │ ---            ┆ la…            ┆ .t…            ┆ ice.…         ┆ ice.…         ┆ atur…         │
        │ str            ┆ ---            ┆ ---            ┆ ---           ┆ ---           ┆ ---           │
        │                ┆ str            ┆ str            ┆ str           ┆ str           ┆ str           │
        ╞════════════════╪════════════════╪════════════════╪═══════════════╪═══════════════╪═══════════════╡
        │ A123           ┆ true           ┆ Mechanical     ┆ USD           ┆ 129.99        ┆ RGB Lighting  │
        │                ┆                ┆ Keyboard       ┆               ┆               ┆               │
        │ A123           ┆ true           ┆ Mechanical     ┆ USD           ┆ 129.99        ┆ Hot-swappable │
        │                ┆                ┆ Keyboard       ┆               ┆               ┆ switches      │
        │ A123           ┆ true           ┆ Mechanical     ┆ USD           ┆ 129.99        ┆ Aluminum      │
        │                ┆                ┆ Keyboard       ┆               ┆               ┆ frame         │
        │ B456           ┆ false          ┆ Noise          ┆ USD           ┆ 299.00        ┆ ANC           │
        │                ┆                ┆ Cancelling     ┆               ┆               ┆               │
        │                ┆                ┆ Headphones     ┆               ┆               ┆               │
        │ B456           ┆ false          ┆ Noise          ┆ USD           ┆ 299.00        ┆ Bluetooth 5.0 │
        │                ┆                ┆ Cancelling     ┆               ┆               ┆               │
        │                ┆                ┆ Headphones     ┆               ┆               ┆               │
        └────────────────┴────────────────┴────────────────┴───────────────┴───────────────┴───────────────┘



    """

    # --- Internal: recursively explode lists + unnest structs ---
    def _fully_flatten(df: pl.DataFrame) -> pl.DataFrame:
        while True:
            list_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.List]
            struct_cols = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Struct]

            if not list_cols and not struct_cols:
                break

            for col in list_cols:
                df = df.explode(col)

            for col in struct_cols:
                fields = df[col].struct.fields
                df = df.unnest(col)
                df = df.rename({f: f"{col}.{f}" for f in fields})

        return df

    # --- Load XML ---
    # Support string/bytes XML payloads or file paths (Path or string)
    if isinstance(xml_input, (bytes, bytearray)):
        is_string = xml_input.lstrip().startswith(b"<")
    elif isinstance(xml_input, str):
        is_string = xml_input.strip().startswith("<")
    elif isinstance(xml_input, Path):
        is_string = False
    else:
        raise TypeError("xml_input must be str, bytes, or pathlib.Path")

    if is_string:
        root = ET.fromstring(xml_input)
    else:
        tree = ET.parse(str(xml_input) if isinstance(xml_input, Path) else xml_input)
        root = tree.getroot()

    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    # --- Flatten element recursively ---
    def flatten_element(element, parent_path=""):
        path_prefix = (
            f"{parent_path}.{strip_ns(element.tag)}"
            if parent_path else strip_ns(element.tag)
        )
        data = {}

        # Attributes
        if include_attributes:
            for k, v in element.attrib.items():
                data[f"{path_prefix}.{strip_ns(k)}"] = v

        # Text if no children
        text = element.text.strip() if element.text else None
        if text and len(element) == 0:
            data[f"{path_prefix}.text"] = text

        # Children
        children_by_tag = {}
        for child in element:
            tag = strip_ns(child.tag)
            children_by_tag.setdefault(tag, []).append(child)

        for tag, siblings in children_by_tag.items():
            if len(siblings) == 1:
                data.update(flatten_element(siblings[0], parent_path=path_prefix))
            else:
                data[f"{path_prefix}.{tag}"] = [
                    flatten_element(s, parent_path="") for s in siblings
                ]

        return data

    # --- Determine record nodes ---
    if record_path:
        parts = record_path.strip(".").split(".")

        # NEW FIX: allow the root to be listed in record_path
        root_tag = strip_ns(root.tag)
        if parts and parts[0] == root_tag:
            parts = parts[1:]

        parent_parts = parts[:-1]
        record_tag = parts[-1]

        # Navigate to parent nodes
        parent_nodes = [root]
        for p in parent_parts:
            next_nodes = []
            for node in parent_nodes:
                next_nodes.extend(node.findall(f"./{p}"))
            parent_nodes = next_nodes

        if not parent_nodes:
            raise ValueError(f"Parent path '{'.'.join(parent_parts)}' not found in XML.")

        # Extract records
        records = []
        for parent in parent_nodes:
            parent_data = {}

            if include_attributes:
                for k, v in parent.attrib.items():
                    parent_data[f"{strip_ns(parent.tag)}.{strip_ns(k)}"] = v

            text = parent.text.strip() if parent.text else None
            if text:
                parent_data[f"{strip_ns(parent.tag)}.text"] = text

            record_nodes = parent.findall(f".//{record_tag}")
            for record in record_nodes:
                record_data = flatten_element(record, parent_path="")
                merged = {**parent_data, **record_data}
                records.append(merged)

    else:
        # No record path → flatten entire root as one record
        records = [flatten_element(root, parent_path="")]

    # --- Wrap primitives if strict=False ---
    def wrap_primitives(obj):
        if isinstance(obj, dict):
            return {k: wrap_primitives(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if all(not isinstance(i, dict) and not isinstance(i, list) for i in obj):
                return obj
            return [wrap_primitives(i) for i in obj]
        else:
            return [obj]

    if not strict:
        records = [wrap_primitives(r) for r in records]

    df = pl.from_dicts(records)

    if flatten:
        df = _fully_flatten(df)

    return df


def write_schema(schema: Union[pl.DataFrame, pl.Schema], file: Union[str, Path]):
    """
    Save a Polars schema to a JSON file.

    Parameters
    ----------
    schema : DataFrame | Schema
        The schema source.
    file : str | Path
        Output JSON file.
    """

    if isinstance(schema, pl.DataFrame):
        schema = schema.schema

    stringified_values = [str(value) for value in schema.dtypes()]
    schema_dict = dict(zip(schema.names(), stringified_values))

    with open(file, "w") as f:
        json.dump(schema_dict, f)


def read_schema(file: Union[str, Path]):
    """
    Load a JSON schema file and return a Polars Schema object.

    Parameters
    ----------
    file : str | Path
        Input JSON file.

    Returns
    -------
    pl.Schema
    """

    with open(file, "r") as f:
        schema = json.load(f)

    schema_dict = {}
    for k, v in schema.items():
        try:
            schema_dict[k] = getattr(pl, v)
        except AttributeError:
            raise ValueError(f"Invalid type {v} for column {k}")

    return pl.Schema(schema_dict)
