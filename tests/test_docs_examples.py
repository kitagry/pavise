import contextlib
import io
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import CodeType
from typing import Literal

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = sorted([*ROOT.glob("docs/**/*.rst"), ROOT / "README.md"])

DATETIME_MODULE_USE = re.compile(r"\bdatetime\.(?:date|datetime|time|timedelta|timezone)\b")
IMPORT_DATETIME = re.compile(r"^\s*import datetime\b", re.MULTILINE)

# Markers are comments so they never show up in the rendered docs:
#   RST:      ``.. example-setup`` (indented hidden code), ``.. example-skip``,
#             ``.. example-raises: ExceptionName``, ``.. example-output``
#   Markdown: ``<!-- example-setup ... -->``, ``<!-- example-skip -->``,
#             ``<!-- example-raises: ExceptionName -->``, ``<!-- example-output -->``
# ``example-output`` goes right before a text block holding what the previous code block
# prints (or ``ExceptionName: message`` for a raised exception). It is opt-in because a
# text block is not always the output of the block right above it.
_RST_INDENTED = r"\n\n((?:(?:[ \t]+.*)?\n)+)"
RST_EXAMPLE = re.compile(
    r"^(?:\.\. code-block:: python(?P<code>" + _RST_INDENTED + r")"
    r"|\.\. example-setup(?P<setup>" + _RST_INDENTED + r")"
    r"|\.\. example-skip$"
    r"|\.\. example-raises: (?P<raises>\w+)$"
    r"|\.\. example-output\n\n\.\. code-block:: text(?P<output>" + _RST_INDENTED + r")"
    r"|(?P<stray_output>\.\. example-output)$)",
    re.MULTILINE,
)
MD_EXAMPLE = re.compile(
    r"^(?:```python\n(?P<code>.*?)^```"
    r"|<!-- example-setup\n(?P<setup>.*?)^-->"
    r"|<!-- example-skip -->$"
    r"|<!-- example-raises: (?P<raises>\w+) -->$"
    r"|<!-- example-output -->\n```text\n(?P<output>.*?)^```"
    r"|(?P<stray_output><!-- example-output -->)$)",
    re.MULTILINE | re.DOTALL,
)

Mode = Literal["run", "skip", "raises"]


@dataclass
class Example:
    code: str
    mode: Mode = "run"
    raises: str | None = None
    output: str | None = None


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
        elif m.group("output") is not None:
            if not examples or examples[-1].mode == "skip":
                raise ValueError("example-output needs a preceding block that runs")
            examples[-1].output = _normalize(m.group("output"))
        elif m.group("stray_output") is not None:
            raise ValueError("example-output must be followed directly by a text block")
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
        name = f"{doc} (example {i})"
        code = compile(example.code, name, "exec")
        stdout = io.StringIO()
        # Capture only when compared, so pytest still reports what a failing block printed.
        capture = (
            contextlib.redirect_stdout(stdout)
            if example.output is not None
            else contextlib.nullcontext()
        )
        with capture:
            if example.mode == "run":
                exec(code, namespace)
            else:
                _exec_expecting(code, namespace, example.raises, name)
        if example.output is not None:
            actual = _normalize(stdout.getvalue())
            assert actual == example.output, (
                f"{name}: documented output differs from the actual output:\n{actual}"
            )


def _exec_expecting(
    code: CodeType, namespace: dict[str, object], raises: str | None, name: str
) -> None:
    try:
        exec(code, namespace)
    except Exception as e:
        if type(e).__name__ != raises:
            raise
        print(f"{type(e).__name__}: {e}")
    else:
        raise AssertionError(f"{name}: expected {raises} but nothing was raised")


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


def test_rst_examples_attach_output_to_previous_code_block():
    text = (
        ".. code-block:: python\n\n   print('hi')\n\n"
        "Output:\n\n"
        ".. example-output\n\n"
        ".. code-block:: text\n\n   hi\n"
    )
    examples = _rst_examples(text)
    assert [(e.code, e.output) for e in examples] == [("print('hi')\n", "hi\n")]


def test_md_examples_attach_output_to_previous_code_block():
    text = "```python\nprint('hi')\n```\n\n<!-- example-output -->\n```text\nhi\n```\n"
    examples = _parse_examples(text, MD_EXAMPLE)
    assert [(e.code, e.output) for e in examples] == [("print('hi')\n", "hi\n")]


@pytest.mark.parametrize(
    "text",
    [
        # The marker must be followed by a ``text`` block, or the output would go unchecked.
        ".. code-block:: python\n\n   print(1)\n\n"
        ".. example-output\n\n.. code-block:: console\n\n   1\n",
        ".. code-block:: python\n\n   print(1)\n\n"
        ".. example-output\n.. code-block:: text\n\n   1\n",
        # Output of a skipped block is never compared.
        ".. example-skip\n\n.. code-block:: python\n\n   print(1)\n\n"
        ".. example-output\n\n.. code-block:: text\n\n   2\n",
        ".. example-output\n\n.. code-block:: text\n\n   1\n",
    ],
)
def test_rst_examples_reject_output_marker_that_would_go_unchecked(text: str):
    with pytest.raises(ValueError, match="example-output"):
        _rst_examples(text)


def test_md_examples_reject_output_marker_without_text_block():
    text = "```python\nprint(1)\n```\n\n<!-- example-output -->\n```console\n1\n```\n"
    with pytest.raises(ValueError, match="example-output"):
        _parse_examples(text, MD_EXAMPLE)


def test_run_examples_fails_when_printed_output_differs():
    with pytest.raises(AssertionError, match="hello"):
        _run_examples([Example("print('hello')\n", output="bye\n")], "doc.rst")


def test_run_examples_accepts_matching_printed_output():
    _run_examples([Example("print('hello')\n", output="hello\n")], "doc.rst")


def test_run_examples_compares_expected_exception_with_output():
    example = Example(
        "raise ValueError('bad value')\n",
        mode="raises",
        raises="ValueError",
        output="ValueError: bad value\n",
    )
    _run_examples([example], "doc.rst")


def test_run_examples_fails_when_exception_message_differs_from_output():
    example = Example(
        "raise ValueError('bad value')\n",
        mode="raises",
        raises="ValueError",
        output="ValueError: other\n",
    )
    with pytest.raises(AssertionError, match="bad value"):
        _run_examples([example], "doc.rst")


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
