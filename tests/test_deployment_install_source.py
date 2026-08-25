"""A deployment must install ZipperGen from wherever this copy came from.

A deployment builds its own Python environment, so it has to name ZipperGen as
something installable. Naming a version works only once the package is on an
index. While it is unpublished, an operator who follows the documented install
-- from Git -- got a working ``zg`` whose very first ``zg deploy`` failed,
because the deployment asked an index for a version that does not exist there.

An install records its own origin in ``direct_url.json``. These tests cover
every shape an install can have, because the failure was exactly a shape that
was never exercised: development checkouts worked, so installed copies were
assumed to.
"""

import json
from pathlib import Path

from zippergen import deployment_environment as de

import pytest


GIT_URL = "https://github.com/zippergen-io/zippergen.git"
COMMIT = "5984302be13d6bdad361a52c4112105fd20359a9"


@pytest.fixture
def not_a_checkout(monkeypatch, tmp_path):
    """Make the module look like an installed copy, not a source tree."""

    monkeypatch.setattr(
        de, "Path", _PathWithoutPyproject(tmp_path), raising=True
    )
    return tmp_path


class _PathWithoutPyproject:
    """A ``Path`` whose ``pyproject.toml`` probe is always negative."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def __call__(self, *args, **kwargs):
        return Path(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(Path, name)


def _origin(monkeypatch, payload):
    """Feed one ``direct_url.json`` body to the origin reader."""

    class _Distribution:
        @staticmethod
        def from_name(_name):
            class _Found:
                @staticmethod
                def read_text(_file):
                    return None if payload is None else json.dumps(payload)

            return _Found()

    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "Distribution", _Distribution)


def test_a_git_install_names_the_commit_it_came_from(monkeypatch) -> None:
    _origin(monkeypatch, {"url": GIT_URL, "vcs_info": {"vcs": "git", "commit_id": COMMIT}})
    assert de._installed_zippergen_origin() == f"git+{GIT_URL}@{COMMIT}"


def test_a_git_install_without_a_commit_still_names_the_repository(
    monkeypatch,
) -> None:
    _origin(monkeypatch, {"url": GIT_URL, "vcs_info": {"vcs": "git"}})
    assert de._installed_zippergen_origin() == f"git+{GIT_URL}"


def test_a_local_install_names_the_directory_it_came_from(monkeypatch) -> None:
    _origin(monkeypatch, {"url": "file:///Users/someone/zippergen"})
    assert de._installed_zippergen_origin() == "/Users/someone/zippergen"


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"url": ""}, "not-an-object-at-all"],
)
def test_an_unusable_record_is_no_origin_rather_than_a_wrong_one(
    monkeypatch, payload
) -> None:
    _origin(monkeypatch, payload)
    assert de._installed_zippergen_origin() is None


def test_a_source_checkout_installs_from_the_checkout() -> None:
    """The development case, which always worked and must keep working."""

    requirement = de._zippergen_install_requirement()
    assert Path(requirement) == Path(de.__file__).resolve().parents[2]
    assert not requirement.startswith("zippergen==")


def test_a_source_checkout_carries_its_extras() -> None:
    requirement = de._zippergen_install_requirement(extras=("google",))
    assert requirement.endswith("[google]")


def test_an_installed_copy_deploys_from_its_own_origin(monkeypatch) -> None:
    """The bug: this used to become ``zippergen==<version>`` and fail."""

    monkeypatch.setattr(
        de, "_installed_zippergen_origin", lambda: f"git+{GIT_URL}@{COMMIT}"
    )
    monkeypatch.setattr(
        de.Path, "exists", lambda _self: False, raising=False
    )
    requirement = de._zippergen_install_requirement()
    assert requirement == f"git+{GIT_URL}@{COMMIT}"
    assert "zippergen==" not in requirement


def test_an_installed_copy_carries_its_extras_with_the_origin(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        de, "_installed_zippergen_origin", lambda: f"git+{GIT_URL}@{COMMIT}"
    )
    monkeypatch.setattr(
        de.Path, "exists", lambda _self: False, raising=False
    )
    requirement = de._zippergen_install_requirement(extras=("google",))
    assert requirement == f"zippergen[google] @ git+{GIT_URL}@{COMMIT}"


def test_an_installed_copy_with_no_origin_falls_back_to_a_version(
    monkeypatch,
) -> None:
    """Correct once published, and the only case where a version is right."""

    monkeypatch.setattr(de, "_installed_zippergen_origin", lambda: None)
    monkeypatch.setattr(
        de.Path, "exists", lambda _self: False, raising=False
    )
    assert de._zippergen_install_requirement().startswith("zippergen==")
