from typing import Union, Optional, List
import xml.etree.ElementTree as ET
import polars as pl

def xml_normalize(
    xml_input: Union[str, bytes],
    record_path: Optional[str] = None,
    include_attributes: bool = True,
    flatten: bool = False,
    strict: bool = True,
) -> pl.DataFrame:
    """
    Flatten XML data into a Polars DataFrame.

    Parameters:
    - xml_input: XML string or file path
    - record_path: dot-separated path to record nodes (e.g., "channel.item")
    - include_attributes: include XML attributes
    - flatten: recursively explode lists and unnest structs
    - strict: True -> Polars raises on type mismatch
              False -> wrap repeated primitives in lists
    """

    # --- Load XML ---
    if xml_input.strip().startswith("<"):
        root = ET.fromstring(xml_input)
    else:
        tree = ET.parse(xml_input)
        root = tree.getroot()

    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    # --- Flatten element recursively ---
    def flatten_element(element, parent_path=""):
        path_prefix = f"{parent_path}.{strip_ns(element.tag)}" if parent_path else strip_ns(element.tag)
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
                data[f"{path_prefix}.{tag}"] = [flatten_element(s, parent_path="") for s in siblings]

        return data

    # --- Determine record nodes ---
    if record_path:
        parts = record_path.strip(".").split(".")
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
            # Parent metadata: attributes + text only
            parent_data = {}
            if include_attributes:
                for k, v in parent.attrib.items():
                    parent_data[f"{strip_ns(parent.tag)}.{strip_ns(k)}"] = v
            text = parent.text.strip() if parent.text else None
            if text:
                parent_data[f"{strip_ns(parent.tag)}.text"] = text

            # Record nodes under parent recursively
            record_nodes = parent.findall(f".//{record_tag}")
            for record in record_nodes:
                record_data = flatten_element(record, parent_path="")
                merged = {**parent_data, **record_data}
                records.append(merged)

    else:
        # No record path → flatten root as single record
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

    # --- Convert to Polars ---
    df = pl.from_dicts(records)

    # --- Optional recursive full flatten ---
    if flatten:
        df = _fully_flatten(df)

    return df


def _fully_flatten(df: pl.DataFrame) -> pl.DataFrame:
    """Recursively explode lists and unnest structs."""
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
