import json
from typing import Union
import polars as pl

def write_schema(schema: Union[pl.DataFrame, pl.Schema], file: str):
    "Saves a Polars schema a JSON file"
    if isinstance(schema, pl.DataFrame):
        schema = schema.schema

    stringified_values = [str(value) for value in schema.dtypes()]
    schema_dict = dict(zip(schema.names(), stringified_values))

    with open(file, "w") as f:
        json.dump(schema_dict, f)
    return

def read_schema(file: str):
    "Opens a JSON Schema file and return a Polars Schema object"
    with open(file, "r") as f:
        schema = json.load(f)
    
    schema_dict = {}
    for k, v in schema.items():
        try:
            schema_dict[k] = getattr(pl, v)
        except AttributeError:
            raise ValueError(f"Invalid type {v} for column {k}")
    
    schema_object = pl.Schema(schema_dict)
    return schema_object