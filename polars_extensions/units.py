import polars as pl

@pl.api.register_expr_namespace("unit_ext")
class UnitExtensionNamespace:
    """
    A Polars expression namespace for unit conversions.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    # ---------------------- TEMPERATURE ----------------------
    def fahrenheit_to_celsius(self) -> pl.Expr:
        """
        Convert Fahrenheit to Celsius.
        
        Examples
        --------
        .. code-block:: python

            import polars as pl
            import polars_extensions as plx

            df = pl.DataFrame({
                "temp_F": [32, 68, 104],
            })
            df.with_columns([
                pl.col("temp_F").unit_ext.fahrenheit_to_celsius().alias("temp_C"),
            ])

        .. code-block:: text

            shape: (3, 2)
            ┌────────┬────────┐
            │ temp_F ┆ temp_C │
            │ ---    ┆ ---    │
            │ i64    ┆ f64    │
            ╞════════╪════════╡
            │ 32     ┆ 0.0    │
            │ 68     ┆ 20.0   │
            │ 104    ┆ 40.0   │
            └────────┴────────┘
        
        """
        return (self._expr - 32) * 5 / 9

    def celsius_to_fahrenheit(self) -> pl.Expr:
        """Convert Celsius to Fahrenheit."""
        return (self._expr * 9 / 5) + 32

    def celsius_to_kelvin(self) -> pl.Expr:
        """Convert Celsius to Kelvin."""
        return self._expr + 273.15

    def kelvin_to_celsius_(self) -> pl.Expr:
        """Convert Kelvin to Celsius."""
        return self._expr - 273.15

    def kelvin_to_rankine(self) -> pl.Expr:
        """Convert Kelvin to Rankine."""
        return self._expr * 9 / 5

    def rankine_to_kelvin(self) -> pl.Expr:
        """Convert Rankine to Kelvin."""
        return self._expr * 5 / 9

    # ---------------------- DISTANCE ----------------------
    def yards_to_meters(self) -> pl.Expr:
        """
        Convert yards to meters.
        
        Examples
        --------
        .. code-block:: python
        
            import polars as pl
            import polars_extensions as plx

            df = pl.DataFrame({
                "yards": [10, 50, 100]
            })

            df.with_columns([
                pl.col("yards").unit_ext.yards_to_meters().alias("meters")
            ])

        
        .. code-block:: text

            shape: (3, 2)
            ┌───────┬────────┐
            │ yards ┆ meters │
            │ ---   ┆ ---    │
            │ i64   ┆ f64    │
            ╞═══════╪════════╡
            │ 10    ┆ 9.144  │
            │ 50    ┆ 45.72  │
            │ 100   ┆ 91.44  │
            └───────┴────────┘

        """
        return self._expr * 0.9144

    def meters_to_yards(self) -> pl.Expr:
        """Convert meters to yards."""
        return self._expr / 0.9144

    def meters_to_feet(self) -> pl.Expr:
        """Convert meters to feet."""
        return self._expr * 3.28084

    def centimeters_to_inches(self) -> pl.Expr:
        """Convert centimeters to inches."""
        return self._expr / 2.54

    def meters_to_kilometers(self) -> pl.Expr:
        """Convert meters to kilometers."""
        return self._expr / 1000

    def kilometers_to_miles(self) -> pl.Expr:
        """Convert kilometers to miles."""
        return self._expr / 1.60934

    def meters_to_nautical_miles(self) -> pl.Expr:
        """Convert meters to nautical miles."""
        return self._expr / 1852

    def meters_to_light_years(self) -> pl.Expr:
        """Convert meters to light years."""
        return self._expr / 9.4607e15

    # ---------------------- WEIGHT / MASS ----------------------
    def pounds_to_kilograms(self) -> pl.Expr:
        """Convert pounds to kilograms."""
        return self._expr * 0.45359237

    def kilograms_to_pounds(self) -> pl.Expr:
        """Convert kilograms to pounds."""
        return self._expr / 0.45359237

    def kilograms_to_grams(self) -> pl.Expr:
        """Convert kilograms to grams."""
        return self._expr * 1000

    def grams_to_ounces(self) -> pl.Expr:
        """Convert grams to ounces."""
        return self._expr / 28.3495

    def kilograms_to_stones(self) -> pl.Expr:
        """Convert kilograms to stones."""
        return self._expr / 6.35029

    # ---------------------- VOLUME ----------------------
    def gallons_to_liters(self) -> pl.Expr:
        """Convert gallons to liters."""
        return self._expr * 3.78541

    def liters_to_gallons(self) -> pl.Expr:
        """Convert liters to gallons."""
        return self._expr / 3.78541

    def liters_to_milliliters(self) -> pl.Expr:
        """Convert liters to milliliters."""
        return self._expr * 1000

    def liters_to_cubic_meters(self) -> pl.Expr:
        """Convert liters to cubic meters."""
        return self._expr / 1000

    def milliliters_to_fluid_ounces(self) -> pl.Expr:
        """Convert milliliters to US fluid ounces."""
        return self._expr / 29.5735

    # ---------------------- ENERGY ----------------------
    def calories_to_joules(self) -> pl.Expr:
        """Convert calories to joules."""
        return self._expr * 4.184

    def joules_to_calories(self) -> pl.Expr:
        """Convert joules to calories."""
        return self._expr / 4.184

    def joules_to_kwh(self) -> pl.Expr:
        """Convert joules to kilowatt-hours."""
        return self._expr / 3.6e6

    def joules_to_btus(self) -> pl.Expr:
        """Convert joules to BTUs."""
        return self._expr / 1055.06

    def joules_to_therms(self) -> pl.Expr:
        """Convert joules to therms."""
        return self._expr / 1.055e8

    # ---------------------- SPEED ----------------------
    def kph_to_mps(self) -> pl.Expr:
        """Convert kilometers per hour to meters per second."""
        return self._expr / 3.6

    def mps_to_kph(self) -> pl.Expr:
        """Convert meters per second to kilometers per hour."""
        return self._expr * 3.6

    def kph_to_mph(self) -> pl.Expr:
        """Convert kilometers per hour to miles per hour."""
        return self._expr / 1.60934

    def mps_to_knots(self) -> pl.Expr:
        """Convert meters per second to knots."""
        return self._expr * 1.94384

    # ---------------------- AREA ----------------------
    def sq_feet_to_sq_meters(self) -> pl.Expr:
        """Convert square feet to square meters."""
        return self._expr * 0.092903

    def sq_meters_to_sq_feet(self) -> pl.Expr:
        """Convert square meters to square feet."""
        return self._expr / 0.092903

    def sq_meters_to_acres(self) -> pl.Expr:
        """Convert square meters to acres."""
        return self._expr / 4046.86

    def sq_meters_to_hectares(self) -> pl.Expr:
        """Convert square meters to hectares."""
        return self._expr / 10000

    # ---------------------- PRESSURE ----------------------
    def psi_to_pascals(self) -> pl.Expr:
        """Convert psi to Pascals."""
        return self._expr * 6894.76

    def pascals_to_bar(self) -> pl.Expr:
        """Convert Pascals to bar."""
        return self._expr / 1e5

    def pascals_to_atm(self) -> pl.Expr:
        """Convert Pascals to atmospheres."""
        return self._expr / 101325

    def pascals_to_torr(self) -> pl.Expr:
        """Convert Pascals to torr."""
        return self._expr * 0.00750062

    # ---------------------- TIME ----------------------
    def sec_to_minutes(self) -> pl.Expr:
        """Convert seconds to minutes."""
        return self._expr / 60

    def sec_to_hours(self) -> pl.Expr:
        """Convert seconds to hours."""
        return self._expr / 3600

    def hours_to_days(self) -> pl.Expr:
        """Convert hours to days."""
        return self._expr / 24

    def days_to_weeks(self) -> pl.Expr:
        """Convert days to weeks."""
        return self._expr / 7

    def days_to_years(self) -> pl.Expr:
        """Convert days to years (approx, 365 days)."""
        return self._expr / 365

    # ---------------------- DIGITAL / DATA ----------------------
    def bytes_to_kilobytes(self) -> pl.Expr:
        """Convert bytes to kilobytes."""
        return self._expr / 1024

    def bytes_to_megabytes(self) -> pl.Expr:
        """Convert bytes to megabytes."""
        return self._expr / (1024**2)

    def bytes_to_gigabytes(self) -> pl.Expr:
        """Convert bytes to gigabytes."""
        return self._expr / (1024**3)

    def bytes_to_terabytes(self) -> pl.Expr:
        """Convert bytes to terabytes."""
        return self._expr / (1024**4)

    def bytes_to_bits(self) -> pl.Expr:
        """Convert bytes to bits."""
        return self._expr * 8

