from pavise import exceptions
from pavise.exceptions import ValidationError


def test_validation_error_inherits_pavise_error():
    assert issubclass(ValidationError, exceptions.PaviseError)


def test_patrol_error_is_alias_of_pavise_error():
    assert exceptions.PatrolError is exceptions.PaviseError


def test_validation_error_names_the_nested_path():
    """An error inside a List or Struct column names the path, but keeps the column name"""
    error = ValidationError("expected str, got Int64", column_name="history", path="history[].zip")
    assert str(error) == "Column 'history[].zip': expected str, got Int64"
    assert error.column_name == "history"
    assert error.path == "history[].zip"


def test_validation_error_with_samples_names_the_nested_path():
    error = ValidationError.new_with_samples(
        "tags", "is non-optional but contains null values", [(0, ["a", None])], 1, path="tags[]"
    )
    assert str(error).startswith("Column 'tags[]': is non-optional")
    assert error.path == "tags[]"


def test_validation_error_path_defaults_to_none():
    assert ValidationError("missing", column_name="a").path is None
