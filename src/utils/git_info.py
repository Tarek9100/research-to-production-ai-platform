import os
import subprocess


def _run_git(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None


def get_git_commit():
    container_commit = os.getenv(
        "SOURCE_GIT_COMMIT"
    )

    if container_commit:
        return container_commit

    return _run_git(
        "rev-parse",
        "HEAD",
    )


def get_git_branch():
    container_branch = os.getenv(
        "SOURCE_GIT_BRANCH"
    )

    if container_branch:
        return container_branch

    return _run_git(
        "branch",
        "--show-current",
    )


def is_git_dirty():
    container_dirty = os.getenv(
        "SOURCE_GIT_DIRTY"
    )

    if container_dirty is not None:
        return (
            container_dirty
            .strip()
            .lower()
            == "true"
        )

    status = _run_git(
        "status",
        "--porcelain",
    )

    if status is None:
        return None

    return bool(status)
