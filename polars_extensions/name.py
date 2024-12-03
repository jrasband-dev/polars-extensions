import polars as pl
import re


@pl.api.register_dataframe_namespace("name_ext")
class NameExtensionNameSpace:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def to_pascal_case(self) -> pl.DataFrame:
        def _to_pascal_case(name: str) -> str:
            return ''.join(word.capitalize() for word in re.sub(r'[_\s]+', ' ', name).split())

        columns = self._df.columns
        new_columns = {col: _to_pascal_case(col) for col in columns}
        return self._df.rename(new_columns)


    def to_snake_case(self) -> pl.DataFrame:
        def _to_snake_case(name: str) -> str:
            return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        new_columns = {col: _to_snake_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)
    
    def to_camel_case(self) -> pl.DataFrame:
        def _to_camel_case(name: str) -> str:
            words = re.sub(r'[_\s]+', ' ', name).split()
            return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
        
        new_columns = {col: _to_camel_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)

    def to_pascal_snake_case(self) -> pl.DataFrame:
        def _to_pascal_snake_case(name: str) -> str:
            words = re.sub(r'[_\s]+', ' ', name).split()
            return '_'.join(word.capitalize() for word in words)
        new_columns = {col: _to_pascal_snake_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)
    

    def to_kebeb_case(self) -> pl.DataFrame:
        def _to_kebeb_case(name: str) -> str:
            return re.sub(r'(?<!^)(?=[A-Z])', '-', name).lower().replace('_', '-')
        
        new_columns = {col: _to_kebeb_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)

    
    def to_upper_snake_case(self) -> pl.DataFrame:
        def _to_upper_snake_case(name: str) -> str:
            return re.sub(r'(?<!^)(?=[A-Z])', '_', name).upper().replace('-', '_')
        
        new_columns = {col: _to_upper_snake_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)


    def to_train_case(self) -> pl.DataFrame:
        def _to_train_case(name: str) -> str:
            return '-'.join(word.capitalize() for word in re.sub(r'[_\s]+', ' ', name).split())
        
        new_columns = {col: _to_train_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)
