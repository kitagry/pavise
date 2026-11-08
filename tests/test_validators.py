from typing import Annotated, Any, Optional

import pytest

from pavise.validators import Custom, Ge, Gt, In, Le, Lt, MaxLen, MinLen, Range, Regex, Unique


def _is_positive(value: Any) -> bool:
    return value > 0


VALIDATORS = [
    lambda: Range(0, 10),
    lambda: Ge(0),
    lambda: Gt(0),
    lambda: Le(0),
    lambda: Lt(0),
    lambda: Unique(),
    lambda: In(["a", "b"]),
    lambda: Regex("a+"),
    lambda: MinLen(1),
    lambda: MaxLen(1),
    lambda: Custom(_is_positive, "must be positive"),
]


@pytest.mark.parametrize("make", VALIDATORS)
def test_equal_validators_have_equal_hashes(make: Any) -> None:
    """Python 3.10 hashes Annotated metadata to deduplicate Annotated[T, v] | None"""
    assert hash(make()) == hash(make())


@pytest.mark.parametrize("make", VALIDATORS)
def test_validator_can_be_wrapped_in_optional(make: Any) -> None:
    annotation = Optional[Annotated[int, make()]]  # noqa: UP045 - same union as `X | None`
    assert annotation is not None
