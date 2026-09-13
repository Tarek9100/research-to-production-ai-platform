import os

from src.utils.git_info import (
    get_git_commit,
    get_git_branch,
    is_git_dirty,
)


def test_container_git_metadata_overrides_repository(monkeypatch):
    monkeypatch.setenv(
        "SOURCE_GIT_COMMIT",
        "abc123def456",
    )

    monkeypatch.setenv(
        "SOURCE_GIT_BRANCH",
        "main",
    )

    monkeypatch.setenv(
        "SOURCE_GIT_DIRTY",
        "false",
    )

    assert get_git_commit() == "abc123def456"
    assert get_git_branch() == "main"
    assert is_git_dirty() is False
