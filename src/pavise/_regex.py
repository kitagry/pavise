"""Backend-independent helpers for the Regex validator."""

import re

# Leading global inline flags such as "(?i)" or "(?ms)"
_LEADING_FLAGS = re.compile(r"\A(?:\(\?[aiLmsux]+\))+")


def fullmatch_pattern(pattern: str) -> str:
    """Anchor ``pattern`` so a substring search behaves like ``re.fullmatch``.

    Meant for engines where ``$`` only matches at the end of the text (polars' Rust regex,
    pyarrow's RE2); Python's ``re`` also lets ``$`` match before a trailing newline.

    The group keeps top-level alternation inside the anchors. Leading global flags become the
    group's scoped flags: left in place they would sit mid-pattern, which Python 3.11+ rejects
    (pandas compiles the pattern with ``re`` first), and hoisted in front of the anchors
    ``(?m)`` would turn ``^``/``$`` into line anchors.
    """
    flags_match = _LEADING_FLAGS.match(pattern)
    if not flags_match:
        return f"^(?:{pattern})$"
    letters = dict.fromkeys(c for c in flags_match.group() if c.isalpha())
    return f"^(?{''.join(letters)}:{pattern[flags_match.end() :]})$"
