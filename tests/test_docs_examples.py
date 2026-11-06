import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = sorted([*ROOT.glob("docs/**/*.rst"), ROOT / "README.md"])

RST_BLOCK = re.compile(r"^\.\. code-block:: python\n\n((?:(?:[ \t]+.*)?\n)+)", re.MULTILINE)
MD_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
DATETIME_MODULE_USE = re.compile(r"\bdatetime\.(?:date|datetime|time|timedelta|timezone)\b")
IMPORT_DATETIME = re.compile(r"^\s*import datetime\b", re.MULTILINE)


def _python_blocks(path: Path) -> list[str]:
    text = path.read_text()
    pattern = MD_BLOCK if path.suffix == ".md" else RST_BLOCK
    return [m.group(1) for m in pattern.finditer(text)]


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
