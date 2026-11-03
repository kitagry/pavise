from importlib.metadata import version

import pavise


def test_version_matches_installed_package_metadata():
    assert pavise.__version__ == version("pavise")
