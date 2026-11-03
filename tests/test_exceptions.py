from pavise import exceptions
from pavise.exceptions import ValidationError


def test_validation_error_inherits_pavise_error():
    assert issubclass(ValidationError, exceptions.PaviseError)


def test_patrol_error_is_alias_of_pavise_error():
    assert exceptions.PatrolError is exceptions.PaviseError
