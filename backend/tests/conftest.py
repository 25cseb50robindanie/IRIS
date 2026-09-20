"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    """Run every test from a scratch working directory.

    The backend uses cwd-relative data/ paths (catalog, COGs, crops, FAISS index), so without this
    a test that ingests a scene would write synthetic vectors into the real search index.
    """
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    yield workdir
