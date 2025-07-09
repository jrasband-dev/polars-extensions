import polars as pl
from shapely import wkb, wkt
from shapely.geometry.base import BaseGeometry

@pl.api.register_dataframe_namespace("geo")
class GeometryExtensionNamespace:
    """Geometry utilities for handling WKB, WKT, and coordinate conversion."""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def _geom_to_coords(self, geom: BaseGeometry):
        """Convert any shapely geometry to a nested coordinate list."""
        if geom.geom_type == "Point":
            return list(geom.coords[0])
        elif geom.geom_type in {"LineString", "LinearRing"}:
            return [list(coord) for coord in geom.coords]
        elif geom.geom_type == "Polygon":
            exterior = [list(coord) for coord in geom.exterior.coords]
            interiors = [[list(coord) for coord in ring.coords] for ring in geom.interiors]
            return [exterior] + interiors if interiors else [exterior]
        elif geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
            return [self._geom_to_coords(part) for part in geom.geoms]
        else:
            return None  # Unknown type

    def wkb_to_coords(self, col: str, output_col: str = "coords") -> pl.DataFrame:
        coords = [
            self._geom_to_coords(wkb.loads(bytes.fromhex(val))) if val else None
            for val in self._df[col]
        ]
        return self._df.with_columns(pl.Series(output_col, coords))

    def coords_to_wkb(self, col: str, output_col: str = "wkb") -> pl.DataFrame:
        from shapely.geometry import shape
        wkb_hex = [
            shape(geom).wkb.hex() if geom else None
            for geom in self._df[col]
        ]
        return self._df.with_columns(pl.Series(output_col, wkb_hex))

    def wkt_to_coords(self, col: str, output_col: str = "coords") -> pl.DataFrame:
        coords = [
            self._geom_to_coords(wkt.loads(val)) if val else None
            for val in self._df[col]
        ]
        return self._df.with_columns(pl.Series(output_col, coords))

    def coords_to_wkt(self, col: str, output_col: str = "wkt") -> pl.DataFrame:
        from shapely.geometry import shape
        wkt_strs = [
            shape(geom).wkt if geom else None
            for geom in self._df[col]
        ]
        return self._df.with_columns(pl.Series(output_col, wkt_strs))
