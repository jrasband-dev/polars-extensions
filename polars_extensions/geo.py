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
        """Convert Well-Known Binary to Coordinates

        
        Examples
        --------
        .. code-block:: python

            import polars as pl 
            import polars_extensions as plx

            data = pl.DataFrame({
                "geometry": [
                    [-101.044612343761, 45.139066210329],
                    [-101.044119223429, 48.1390850482555],
                    [-102.044733837176, 43.1389478003816],
                    [-114.04470525049, 43.1385010700204],
                ]
            },schema_overrides={'geometry':pl.Object})

            data.with_columns(pl.col('geometry').geo_ext.coords_to_wkt().alias('wkt'))


        .. code-block:: text

            shape: (4, 2)
            ┌─────────────────────────────────┬─────────────────────────────────┐
            │ geometry                        ┆ wkt                             │
            │ ---                             ┆ ---                             │
            │ object                          ┆ str                             │
            ╞═════════════════════════════════╪═════════════════════════════════╡
            │ [-101.044612343761, 45.1390662… ┆ POINT (-101.044612343761 45.13… │
            │ [-101.044119223429, 48.1390850… ┆ POINT (-101.044119223429 48.13… │
            │ [-102.044733837176, 43.1389478… ┆ POINT (-102.044733837176 43.13… │
            │ [-114.04470525049, 43.13850107… ┆ POINT (-114.04470525049 43.138… │
            └─────────────────────────────────┴─────────────────────────────────┘
        """
        from shapely import wkb

        return self._expr.map_elements(
            lambda x: self._geom_to_coords(wkb.loads(bytes.fromhex(x))) if x else None,
            return_dtype=pl.Object
        )

    def coords_to_wkb(self) -> pl.Expr:
        """Convert Coordinates to Well-Known Binary
    
        Examples
        --------
        .. code-block:: python

            import polars as pl 
            import polars_extensions as plx

            data = pl.DataFrame({
                "geometry": [
                    [-101.044612343761, 45.139066210329],
                    [-101.044119223429, 48.1390850482555],
                    [-102.044733837176, 43.1389478003816],
                    [-114.04470525049, 43.1385010700204],
                ]
            },schema_overrides={'geometry':pl.Object})

            data.with_columns(pl.col('geometry').geo_ext.coords_to_wkb().alias('wkb'))

        .. code-block:: text

            shape: (4, 2)
            ┌─────────────────────────────────┬─────────────────────────────────┐
            │ geometry                        ┆ wkb                             │
            │ ---                             ┆ ---                             │
            │ object                          ┆ str                             │
            ╞═════════════════════════════════╪═════════════════════════════════╡
            │ [-101.044612343761, 45.1390662… ┆ 0101000000e45cbbedda4259c0bdab… │
            │ [-101.044119223429, 48.1390850… ┆ 010100000029706fd9d24259c05bcf… │
            │ [-102.044733837176, 43.1389478… ┆ 010100000083ec4febdc8259c0bc3e… │
            │ [-114.04470525049, 43.13850107… ┆ 010100000019346973dc825cc06d19… │
            └─────────────────────────────────┴─────────────────────────────────┘

        """
        from shapely.geometry import shape

        return self._expr.map_elements(
            lambda x: shape(self._coords_to_geojson(x)).wkb.hex() if x else None,
            return_dtype=pl.String
        )
    def wkt_to_coords(self) -> pl.Expr:
        """Convert Well-Known Text to Coordinates
        
        Examples
        --------
        .. code-block:: python

            import polars as pl 
            import polars_extensions as plx

            data = pl.DataFrame({
                "geometry": [
                    "POINT (-101.044612343761 45.139066210329)",
                    "POINT (-101.044119223429 48.1390850482555)",
                    "POINT (-102.044733837176 43.1389478003816)",
                    "POINT (-114.04470525049 43.1385010700204)",
                ]
            })
            data.with_columns(pl.col('geometry').geo_ext.wkt_to_coords().alias('coords'))

        .. code-block:: text

            shape: (4, 2)
            ┌─────────────────────────────────┬─────────────────────────────────┐
            │ geometry                        ┆ coords                          │
            │ ---                             ┆ ---                             │
            │ str                             ┆ object                          │
            ╞═════════════════════════════════╪═════════════════════════════════╡
            │ POINT (-101.044612343761 45.13… ┆ [-101.044612343761, 45.1390662… │
            │ POINT (-101.044119223429 48.13… ┆ [-101.044119223429, 48.1390850… │
            │ POINT (-102.044733837176 43.13… ┆ [-102.044733837176, 43.1389478… │
            │ POINT (-114.04470525049 43.138… ┆ [-114.04470525049, 43.13850107… │
            └─────────────────────────────────┴─────────────────────────────────┘
        """
        from shapely import wkt

        return self._expr.map_elements(
            lambda x: self._geom_to_coords(wkt.loads(x)) if x else None,
            return_dtype=pl.Object
        )

    def coords_to_wkt(self) -> pl.Expr:
        """Convert Coordinates to Well-Known Text
    
        Examples
        --------
        .. code-block:: python

            import polars as pl 
            import polars_extensions as plx

            data = pl.DataFrame({
                "geometry": [
                    [-101.044612343761, 45.139066210329],
                    [-101.044119223429, 48.1390850482555],
                    [-102.044733837176, 43.1389478003816],
                    [-114.04470525049, 43.1385010700204],
                ]
            },schema_overrides={'geometry':pl.Object})

            data.with_columns(pl.col('geometry').geo_ext.coords_to_wkt().alias('wkb'))  

        .. code-block:: text

            shape: (4, 2)
            ┌─────────────────────────────────┬─────────────────────────────────┐
            │ geometry                        ┆ wkb                             │
            │ ---                             ┆ ---                             │
            │ object                          ┆ str                             │
            ╞═════════════════════════════════╪═════════════════════════════════╡
            │ [-101.044612343761, 45.1390662… ┆ POINT (-101.044612343761 45.13… │
            │ [-101.044119223429, 48.1390850… ┆ POINT (-101.044119223429 48.13… │
            │ [-102.044733837176, 43.1389478… ┆ POINT (-102.044733837176 43.13… │
            │ [-114.04470525049, 43.13850107… ┆ POINT (-114.04470525049 43.138… │
            └─────────────────────────────────┴─────────────────────────────────┘
     
        """
        from shapely.geometry import shape

        return self._expr.map_elements(
            lambda x: shape(self._coords_to_geojson(x)).wkt if x else None,
            return_dtype=pl.String
        )
    

    def wkb_to_wkt(self) ->pl.Expr:
        """Convert Well-Known Binary to Well-Known Text
        
        Examples
        --------
        .. code-block:: python

            import polars as pl 
            import polars_extensions as plx

            data = pl.DataFrame({
                "wkb": [
                    '0101000000e45cbbedda4259c0bdabecebcc914640',
                    '010100000029706fd9d24259c05bcff289cd114840',
                    '010100000083ec4febdc8259c0bc3ea10ac9914540',
                    '010100000019346973dc825cc06d192f67ba914540',
                ]
            })

            data.with_columns(pl.col('wkb').geo_ext.wkb_to_wkt().alias('wkt'))

        .. code-block:: text 

            shape: (4, 2)
            ┌─────────────────────────────────┬─────────────────────────────────┐
            │ wkb                             ┆ wkt                             │
            │ ---                             ┆ ---                             │
            │ str                             ┆ str                             │
            ╞═════════════════════════════════╪═════════════════════════════════╡
            │ 0101000000e45cbbedda4259c0bdab… ┆ POINT (-101.044612343761 45.13… │
            │ 010100000029706fd9d24259c05bcf… ┆ POINT (-101.044119223429 48.13… │
            │ 010100000083ec4febdc8259c0bc3e… ┆ POINT (-102.044733837176 43.13… │
            │ 010100000019346973dc825cc06d19… ┆ POINT (-114.04470525049 43.138… │
            └─────────────────────────────────┴─────────────────────────────────┘
        
        """
        if self._expr is None:
            return None
            
        return self._expr.map_elements(
            lambda x: wkb.loads(x).wkt if x else None,
            return_dtype=pl.String
        )


    def wkt_to_wkb(self,format='raw') -> pl.Expr:
        """Convert Well-Known Text to Well-Known Binary
        
    
        Examples
        --------
        .. code-block:: python

            import polars as pl 
            import polars_extensions as plx

            data = pl.DataFrame({
                "geometry": [
                    "POINT (-101.044612343761 45.139066210329)",
                    "POINT (-101.044119223429 48.1390850482555)",
                    "POINT (-102.044733837176 43.1389478003816)",
                    "POINT (-114.04470525049 43.1385010700204)",
                ]
            })
            data.with_columns(
                pl.col('geometry').geo_ext.wkt_to_wkb().alias('coords'))

        .. code-block:: text 

            shape: (4, 2)
            ┌─────────────────────────────────┬─────────────────────────────────┐
            │ geometry                        ┆ coords                          │
            │ ---                             ┆ ---                             │
            │ str                             ┆ binary                          │
            ╞═════════════════════════════════╪═════════════════════════════════╡
            │ POINT (-101.044612343761 45.13… ┆ b"\x01\x01\x00\x00\x00\xe4\\xb… │
            │ POINT (-101.044119223429 48.13… ┆ b"\x01\x01\x00\x00\x00)po\xd9\… │
            │ POINT (-102.044733837176 43.13… ┆ b"\x01\x01\x00\x00\x00\x83\xec… │
            │ POINT (-114.04470525049 43.138… ┆ b"\x01\x01\x00\x00\x00\x194is\… │
            └─────────────────────────────────┴─────────────────────────────────┘

        """
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