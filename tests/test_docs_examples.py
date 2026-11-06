import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = sorted([*ROOT.glob("docs/**/*.rst"), ROOT / "README.md"])

DATETIME_MODULE_USE = re.compile(r"\bdatetime\.(?:date|datetime|time|timedelta|timezone)\b")
IMPORT_DATETIME = re.compile(r"^\s*import datetime\b", re.MULTILINE)

# Markers are comments so they never show up in the rendered docs:
#   RST:      ``.. example-setup`` (indented hidden code), ``.. example-skip``,
#             ``.. example-raises: ExceptionName``
#   Markdown: ``<!-- example-setup ... -->``, ``<!-- example-skip -->``,
#             ``<!-- example-raises: ExceptionName -->``
_RST_INDENTED = r"\n\n((?:(?:[ \t]+.*)?\n)+)"
RST_EXAMPLE = re.compile(
    r"^(?:\.\. code-block:: python(?P<code>" + _RST_INDENTED + r")"
    r"|\.\. example-setup(?P<setup>" + _RST_INDENTED + r")"
    r"|\.\. example-skip$"
    r"|\.\. example-raises: (?P<raises>\w+)$)",
    re.MULTILINE,
)
MD_EXAMPLE = re.compile(
    r"^(?:```python\n(?P<code>.*?)^```"
    r"|<!-- example-setup\n(?P<setup>.*?)^-->"
    r"|<!-- example-skip -->$"
    r"|<!-- example-raises: (?P<raises>\w+) -->$)",
    re.MULTILINE | re.DOTALL,
)

Mode = Literal["run", "skip", "raises"]


@dataclass
class Example:
    code: str
    mode: Mode = "run"
    raises: str | None = None


def _pattern(path: Path) -> re.Pattern[str]:
    return MD_EXAMPLE if path.suffix == ".md" else RST_EXAMPLE


def _python_blocks(path: Path) -> list[str]:
    matches = _pattern(path).finditer(path.read_text())
    return [m.group("code") for m in matches if m.group("code") is not None]


def _parse_examples(text: str, pattern: re.Pattern[str]) -> list[Example]:
    examples: list[Example] = []
    mode: Mode = "run"
    raises: str | None = None
    for m in pattern.finditer(text):
        if m.group("setup") is not None:
            examples.append(Example(_normalize(m.group("setup"))))
        elif m.group("code") is not None:
            examples.append(Example(_normalize(m.group("code")), mode, raises))
            mode, raises = "run", None
        elif m.group("raises") is not None:
            mode, raises = "raises", m.group("raises")
        else:
            mode = "skip"
    return examples


def _normalize(block: str) -> str:
    return textwrap.dedent(block).strip("\n") + "\n"


def _rst_examples(text: str) -> list[Example]:
    return _parse_examples(text, RST_EXAMPLE)


def _examples(path: Path) -> list[Example]:
    return _parse_examples(path.read_text(), _pattern(path))


def _run_examples(examples: list[Example], doc: str) -> None:
    # Blocks on one page share a namespace, the same way a reader follows the page top to bottom.
    namespace: dict[str, object] = {"__name__": "__doc_example__"}
    for i, example in enumerate(examples):
        if example.mode == "skip":
            continue
        code = compile(example.code, f"{doc} (example {i})", "exec")
        if example.mode == "run":
            exec(code, namespace)
            continue
        try:
            exec(code, namespace)
        except Exception as e:
            if type(e).__name__ != example.raises:
                raise
        else:
            raise AssertionError(
                f"{doc} (example {i}): expected {example.raises} but nothing was raised"
            )


def _blocks_missing_datetime_import() -> list[tuple[str, str]]:
    missing = []
    for path in DOC_FILES:
        for block in _python_blocks(path):
            if DATETIME_MODULE_USE.search(block) and not IMPORT_DATETIME.search(block):
                missing.append((str(path.relative_to(ROOT)), block))
    return missing


def test_rst_python_blocks_are_extracted():
    blocks = _python_blocks(ROOT / "docs/examples/data-validation.rst")
    assert any("import datetime" in block for block in blocks)


def test_python_blocks_import_datetime_when_using_it():
    missing = _blocks_missing_datetime_import()
    assert not missing, "\n\n".join(
        f"{path}: code block uses datetime.* without `import datetime`:\n{block}"
        for path, block in missing
    )


def test_rst_examples_keep_order_of_setup_and_code_blocks():
    text = ".. example-setup\n\n   x = 1\n\nSome prose.\n\n.. code-block:: python\n\n   y = x + 1\n"
    examples = _rst_examples(text)
    assert [(e.code, e.mode) for e in examples] == [("x = 1\n", "run"), ("y = x + 1\n", "run")]


def test_rst_examples_apply_marker_to_following_code_block():
    text = (
        ".. example-raises: ValidationError\n\n"
        ".. code-block:: python\n\n   fail()\n\n"
        ".. example-skip\n\n"
        ".. code-block:: python\n\n   unreachable()\n\n"
        ".. code-block:: python\n\n   ok()\n"
    )
    examples = _rst_examples(text)
    assert [(e.code, e.mode, e.raises) for e in examples] == [
        ("fail()\n", "raises", "ValidationError"),
        ("unreachable()\n", "skip", None),
        ("ok()\n", "run", None),
    ]


def test_run_examples_shares_namespace_between_blocks():
    examples = [Example("x = 1\n"), Example("assert x == 1\n")]
    _run_examples(examples, "doc.rst")


def test_run_examples_fails_when_expected_exception_is_not_raised():
    with pytest.raises(AssertionError, match="ValueError"):
        _run_examples([Example("x = 1\n", mode="raises", raises="ValueError")], "doc.rst")


def test_run_examples_accepts_expected_exception():
    _run_examples([Example("raise ValueError\n", mode="raises", raises="ValueError")], "doc.rst")


def test_run_examples_does_not_execute_skipped_blocks():
    _run_examples([Example("raise RuntimeError\n", mode="skip")], "doc.rst")


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_doc_examples_run(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Examples read and write files like users.csv, so keep them out of the repo.
    monkeypatch.chdir(tmp_path)
    _run_examples(_examples(path), str(path.relative_to(ROOT)))
