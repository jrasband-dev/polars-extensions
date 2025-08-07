import polars as pl
from shapely.geometry.base import BaseGeometry
from shapely import wkb
from shapely import wkt


@pl.api.register_expr_namespace("geo_ext")
class GeometryExtensionNamespace:
    """Geometry utilities for handling WKB, WKT, and coordinate conversion."""

    def __init__(self, expr: pl.Expr):
        self._expr = expr

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
        

    def _coords_to_geojson(self, coords):
        """Infer geometry type from coordinates and return GeoJSON-like dict."""
        if not coords:
            return None
        # Point: [x, y]
        if isinstance(coords[0], (float, int)):
            return {"type": "Point", "coordinates": coords}
        # LineString: [[x, y], ...]
        if isinstance(coords[0], list) and isinstance(coords[0][0], (float, int)):
            return {"type": "LineString", "coordinates": coords}
        # Polygon: [[[x, y], ...], ...]
        if isinstance(coords[0], list) and isinstance(coords[0][0], list):
            return {"type": "Polygon", "coordinates": coords}
        # Multi geometries or GeometryCollection
        # Try MultiPoint
        if all(isinstance(c, list) and isinstance(c[0], (float, int)) for c in coords):
            return {"type": "MultiPoint", "coordinates": coords}
        # Try MultiLineString
        if all(isinstance(c, list) and isinstance(c[0], list) and isinstance(c[0][0], (float, int)) for c in coords):
            return {"type": "MultiLineString", "coordinates": coords}
        # Try MultiPolygon
        if all(isinstance(c, list) and isinstance(c[0], list) and isinstance(c[0][0], list) for c in coords):
            return {"type": "MultiPolygon", "coordinates": coords}
        # Fallback
        return None

    def wkb_to_coords(self) -> pl.Expr:
        from shapely import wkb

        return self._expr.map_elements(
            lambda x: self._geom_to_coords(wkb.loads(bytes.fromhex(x))) if x else None,
            return_dtype=pl.Object
        )

    def coords_to_wkb(self) -> pl.Expr:
        from shapely.geometry import shape

        return self._expr.map_elements(
            lambda x: shape(self._coords_to_geojson(x)).wkb.hex() if x else None,
            return_dtype=pl.String
        )
    def wkt_to_coords(self) -> pl.Expr:
        from shapely import wkt

        return self._expr.map_elements(
            lambda x: self._geom_to_coords(wkt.loads(x)) if x else None,
            return_dtype=pl.Object
        )

    def coords_to_wkt(self) -> pl.Expr:
        from shapely.geometry import shape

        return self._expr.map_elements(
            lambda x: shape(self._coords_to_geojson(x)).wkt if x else None,
            return_dtype=pl.String
        )
    

    def wkb_to_wkt(self) ->pl.Expr:
        if self._expr is None:
            return None
            
        return self._expr.map_elements(
            lambda x: wkb.loads(x).wkt if x else None,
            return_dtype=pl.String
        )


    def wkt_to_wkb(self,format='raw') -> pl.Expr:
        if self._expr is None:
            return None

        elif format == 'hex':
            return self._expr.map_elements(
                lambda x: wkt.loads(x).wkb.hex() if x else None,
                return_dtype=pl.String
            )
        elif format == 'raw':
            return self._expr.map_elements(
                lambda x: wkt.loads(x).wkb if x else None,
                return_dtype=pl.Binary
            )