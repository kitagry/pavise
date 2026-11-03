from typing import Protocol, runtime_checkable

from pavise.checks import dataframe_check


@runtime_checkable
@dataframe_check(lambda df: df["a"] > 0, name="a_positive")
class RuntimeCheckableSchema(Protocol):
    a: int


class HasA:
    a = 1


def test_dataframe_check_does_not_add_protocol_members():
    assert isinstance(HasA(), RuntimeCheckableSchema)
