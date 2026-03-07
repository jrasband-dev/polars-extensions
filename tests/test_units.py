import polars as pl
import pytest
import polars_extensions


@pytest.mark.parametrize(
    "method_name,expected",
    [
        ("fahrenheit_to_celsius", lambda x: (x - 32) * 5 / 9),
        ("celsius_to_fahrenheit", lambda x: (x * 9 / 5) + 32),
        ("celsius_to_kelvin", lambda x: x + 273.15),
        ("kelvin_to_celsius_", lambda x: x - 273.15),
        ("kelvin_to_rankine", lambda x: x * 9 / 5),
        ("rankine_to_kelvin", lambda x: x * 5 / 9),
        ("yards_to_meters", lambda x: x * 0.9144),
        ("meters_to_yards", lambda x: x / 0.9144),
        ("meters_to_feet", lambda x: x * 3.28084),
        ("centimeters_to_inches", lambda x: x / 2.54),
        ("meters_to_kilometers", lambda x: x / 1000),
        ("kilometers_to_miles", lambda x: x / 1.60934),
        ("meters_to_nautical_miles", lambda x: x / 1852),
        ("meters_to_light_years", lambda x: x / 9.4607e15),
        ("pounds_to_kilograms", lambda x: x * 0.45359237),
        ("kilograms_to_pounds", lambda x: x / 0.45359237),
        ("kilograms_to_grams", lambda x: x * 1000),
        ("grams_to_ounces", lambda x: x / 28.3495),
        ("kilograms_to_stones", lambda x: x / 6.35029),
        ("gallons_to_liters", lambda x: x * 3.78541),
        ("liters_to_gallons", lambda x: x / 3.78541),
        ("liters_to_milliliters", lambda x: x * 1000),
        ("liters_to_cubic_meters", lambda x: x / 1000),
        ("milliliters_to_fluid_ounces", lambda x: x / 29.5735),
        ("calories_to_joules", lambda x: x * 4.184),
        ("joules_to_calories", lambda x: x / 4.184),
        ("joules_to_kwh", lambda x: x / 3.6e6),
        ("joules_to_btus", lambda x: x / 1055.06),
        ("joules_to_therms", lambda x: x / 1.055e8),
        ("kph_to_mps", lambda x: x / 3.6),
        ("mps_to_kph", lambda x: x * 3.6),
        ("kph_to_mph", lambda x: x / 1.60934),
        ("mps_to_knots", lambda x: x * 1.94384),
        ("sq_feet_to_sq_meters", lambda x: x * 0.092903),
        ("sq_meters_to_sq_feet", lambda x: x / 0.092903),
        ("sq_meters_to_acres", lambda x: x / 4046.86),
        ("sq_meters_to_hectares", lambda x: x / 10000),
        ("psi_to_pascals", lambda x: x * 6894.76),
        ("pascals_to_bar", lambda x: x / 1e5),
        ("pascals_to_atm", lambda x: x / 101325),
        ("pascals_to_torr", lambda x: x * 0.00750062),
        ("sec_to_minutes", lambda x: x / 60),
        ("sec_to_hours", lambda x: x / 3600),
        ("hours_to_days", lambda x: x / 24),
        ("days_to_weeks", lambda x: x / 7),
        ("days_to_years", lambda x: x / 365),
        ("bytes_to_kilobytes", lambda x: x / 1024),
        ("bytes_to_megabytes", lambda x: x / (1024**2)),
        ("bytes_to_gigabytes", lambda x: x / (1024**3)),
        ("bytes_to_terabytes", lambda x: x / (1024**4)),
        ("bytes_to_bits", lambda x: x * 8),
    ],
)
def test_units_namespace_transformations(method_name: str, expected) -> None:
    value = 10.0
    df = pl.DataFrame({"x": [value, None]})

    out = df.select(getattr(pl.col("x").unit_ext, method_name)().alias("out"))["out"].to_list()

    assert out[1] is None
    assert out[0] == pytest.approx(expected(value), rel=1e-9, abs=1e-12)
