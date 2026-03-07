import polars as pl
import pytest
import polars_extensions

pytest.importorskip("polars_extensions._name_rust")


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "employee_id": [1],
            "FirstName": ["John"],
            "last name": ["Doe"],
            "DateOfHire": ["2020-01-01"],
            "job-title": ["Engineer"],
        }
    )


@pytest.mark.parametrize(
    "method_name,expected_columns",
    [
        (
            "to_pascal_case",
            ["EmployeeId", "Firstname", "LastName", "Dateofhire", "Job-title"],
        ),
        (
            "to_snake_case",
            ["employee_id", "first_name", "last name", "date_of_hire", "job-title"],
        ),
        (
            "to_camel_case",
            ["employeeId", "firstname", "lastName", "dateofhire", "job-title"],
        ),
        (
            "to_pascal_snake_case",
            ["Employee_Id", "Firstname", "Last_Name", "Dateofhire", "Job-title"],
        ),
        (
            "to_kebeb_case",
            ["employee-id", "first-name", "last name", "date-of-hire", "job-title"],
        ),
        (
            "to_upper_snake_case",
            ["EMPLOYEE_ID", "FIRST_NAME", "LAST NAME", "DATE_OF_HIRE", "JOB_TITLE"],
        ),
        (
            "to_train_case",
            ["Employee-Id", "Firstname", "Last-Name", "Dateofhire", "Job-title"],
        ),
    ],
)
def test_name_namespace_case_transformations(
    sample_df: pl.DataFrame,
    method_name: str,
    expected_columns: list[str],
) -> None:
    result = getattr(sample_df.name_ext, method_name)()
    assert result.columns == expected_columns
