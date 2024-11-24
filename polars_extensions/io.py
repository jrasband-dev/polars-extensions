import json 
from typing import IO, Union
from pathlib import Path
import polars as pl



def write_schema(schema: Union[pl.DataFrame, pl.Schema], file: Union[str, Path, IO[str], IO[bytes]]):
    if isinstance(schema, pl.DataFrame):
        schema = schema.schema

    stringified_values = [str(value) for value in schema.dtypes()]
    schema_dict = dict(zip(schema.names(), stringified_values))
    
    with open(file, 'w') as f:
        json.dump(schema_dict, f)
    return

def read_schema(file: str | Path | IO[str] | IO[bytes]):
    f = open(file,'r')
    schema = json.load(f)
    f.close()
    schema_dict = {k: eval(f"pl.{v}") for k, v in schema.items()}
    schema_object = pl.Schema(schema_dict)
    return schema_object