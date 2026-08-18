"""Checks for the intentional package-level public surface."""

import zippergen
from zippergen import (
    actions,
    assistant_backends,
    backends,
    builder,
    connectors,
    deployment,
    formula,
    google_gmail,
    google_sheets,
    human_backends,
    projection,
    runtime,
    semantic,
    sqlite_runner,
    syntax,
    telegram_chat,
    view,
)


PUBLIC_MODULES = (
    syntax,
    actions,
    backends,
    human_backends,
    assistant_backends,
    formula,
    builder,
    projection,
    runtime,
    sqlite_runner,
    deployment,
    connectors,
    google_sheets,
    google_gmail,
    telegram_chat,
    view,
    semantic,
)


def test_package_public_api_is_explicit_and_complete():
    module_exports = [
        name
        for module in PUBLIC_MODULES
        for name in module.__all__
    ]

    assert len(zippergen.__all__) == len(set(zippergen.__all__))
    assert set(zippergen.__all__) == set(module_exports)
    assert all(hasattr(zippergen, name) for name in zippergen.__all__)
