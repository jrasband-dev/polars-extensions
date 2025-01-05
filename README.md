# Polars Extensions

This library is designed to extend the capabilities of polars with functionalities that are not currently found in Polars. 


## Social Proving Ground

Polars is written in Rust, but these extensions are created in pure python. The concept for this package is to create a social proving ground for extensions that could be added to the main polars library. Those who don't know Rust can develop functions here in Python and the best ideas can be passed on to the main polars library for development. 

## Usage

### Case Conventions
```
import polars as pl
from polars_extensions import *

data = pl.read_csv('datasets/employees.csv')
data
```
```
data.name_ext.to_kebeb_case()
```
```
data.name_ext.to_train_case()
```
```
data.name_ext.to_pascal_case()
```

```
data.name_ext.to_snake_case()
```
```
data.name_ext.to_camel_case()
```
```
data.name_ext.to_pascal_snake_case()
```


### Numeric 

```
import polars as pl
df = pl.DataFrame({"numbers": [1, 2, 309, 4, 5]})
df
```
```
import polars_extensions as plx

result = df.with_columns(
    pl.col('numbers').num_ext.to_roman().alias("Roman")
)
result
```

```
new_result = result.with_columns(
    pl.col('Roman').num_ext.from_roman().alias("Decoded")
)
new_result
```


