import re
from collections.abc import Callable

import pytest

from pavise._regex import fullmatch_pattern


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (r"[a-z]+", r"^(?:[a-z]+)$"),
        (r"a|b", r"^(?:a|b)$"),
        (r"(?i)abc", r"^(?i:abc)$"),
        (r"(?i)(?s)a.c", r"^(?is:a.c)$"),
        (r"(?:x)|y", r"^(?:(?:x)|y)$"),
    ],
)
def test_fullmatch_pattern_scopes_leading_flags_inside_anchors(pattern: str, expected: str):
    assert fullmatch_pattern(pattern) == expected


def _search_polars(pattern: str, value: str) -> bool:
    pl = pytest.importorskip("polars")
    return bool(pl.Series([value]).str.contains(pattern).item())


def _search_pyarrow(pattern: str, value: str) -> bool:
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    return bool(pc.match_substring_regex(pa.array([value]), pattern)[0].as_py())


@pytest.mark.parametrize(
    "search",
    [
        pytest.param(_search_polars, id="polars"),
        pytest.param(_search_pyarrow, id="pyarrow"),
    ],
)
@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        (r"[a-z]+", "abc123"),
        (r"[a-z]+", "abc"),
        (r"^a|b$", "ab"),
        (r"(?i)abc", "ABC"),
        (r"(?i)(?s)a.c", "A\nC"),
        (r"(?m)abc", "x\nabc"),
        (r"abc", "abc\n"),
    ],
)
def test_fullmatch_pattern_search_agrees_with_re_fullmatch(
    search: Callable[[str, str], bool], pattern: str, value: str
):
    """Searching with the anchored pattern on each backend's engine agrees with re.fullmatch"""
    assert search(fullmatch_pattern(pattern), value) == (re.fullmatch(pattern, value) is not None)
