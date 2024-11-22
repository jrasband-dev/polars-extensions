import polars as pl
import re

# Registering the namespace for the extension
@pl.api.register_dataframe_namespace("name_ext")
class NameExtensionNameSpace:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    # Single function to convert all column names to PascalCase
    def to_pascal_case(self) -> pl.DataFrame:
        # Helper function to convert a string to PascalCase
        def _to_pascal_case(name: str) -> str:
            return ''.join(word.capitalize() for word in re.sub(r'[_\s]+', ' ', name).split())

        # Get column names from the DataFrame (if it's a DataFrame)
        columns = self._df.columns

        # Create a mapping of old column names to PascalCase
        new_columns = {col: _to_pascal_case(col) for col in columns}

        # Rename the columns using the new mapping
        return self._df.rename(new_columns)


        # Function to convert all column names to snake_case
    def to_snake_case(self) -> pl.DataFrame:
        # Helper function to convert a string to snake_case
        def _to_snake_case(name: str) -> str:
            # Insert an underscore before capital letters and lowercase everything
            return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

        # Create a mapping of old column names to snake_case
        new_columns = {col: _to_snake_case(col) for col in self._df.columns}

        return self._df.rename(new_columns)
    
    def to_camel_case(self) -> pl.DataFrame:
        # Helper function to convert a string to camelCase
        def _to_camel_case(name: str) -> str:
            words = re.sub(r'[_\s]+', ' ', name).split()
            return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

        # Create a mapping of old column names to camelCase
        new_columns = {col: _to_camel_case(col) for col in self._df.columns}

        # Rename the columns using the new mapping
        return self._df.rename(new_columns)

    # Function to convert all column names to Pascal_Snake_Case
    def to_pascal_snake_case(self) -> pl.DataFrame:
        def _to_pascal_snake_case(name: str) -> str:
            # Replace spaces/underscores, capitalize each word, and join with underscores
            words = re.sub(r'[_\s]+', ' ', name).split()
            return '_'.join(word.capitalize() for word in words)
        new_columns = {col: _to_pascal_snake_case(col) for col in self._df.columns}
        return self._df.rename(new_columns)