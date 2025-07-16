import polars as pl



@pl.api.register_expr_namespace("num_ext") 
class NumericExtensionNamespace: 
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    # Roman numeral mappings
    _roman_map = [
        ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),
        ("C", 100), ("XC", 90), ("L", 50), ("XL", 40),
        ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1)
    ]

    def to_roman(self) -> pl.Expr:
        """
        Convert an integer to Roman numerals.
        """
        def convert_to_roman(value: int) -> str:
            if not (0 < value < 4000):
                raise ValueError("Number out of range (1-3999)")

            result = []
            for roman, num in self._roman_map:
                while value >= num:
                    result.append(roman)
                    value -= num
            return ''.join(result)

        # Use map_elements for element-wise operation
        return self._expr.map_elements(convert_to_roman,return_dtype=pl.String)

    def from_roman(self) -> pl.Expr:
        """
        Convert Roman numerals to integers.
        """
        roman_to_value = {roman: value for roman, value in self._roman_map}

        def convert_from_roman(roman: str) -> int:
            i = 0
            total = 0
            while i < len(roman):
                # Check for two-character numeral first
                if i + 1 < len(roman) and roman[i:i+2] in roman_to_value:
                    total += roman_to_value[roman[i:i+2]]
                    i += 2
                else:
                    total += roman_to_value[roman[i]]
                    i += 1
            return total

        # Use map_elements for element-wise operation
        return self._expr.map_elements(convert_from_roman,return_dtype=pl.Int64)
    
    def word_to_number(self) -> pl.Expr:
        from word2number import w2n
        
        return self._expr.map_elements(lambda x: w2n.word_to_num(x) if isinstance(x, str) else x, return_dtype=pl.Int64)
