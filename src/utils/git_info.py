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
    return _run_git(
        "rev-parse",
        "HEAD",
    )


def get_git_branch():
    return _run_git(
        "branch",
        "--show-current",
    )


def is_git_dirty():
    status = _run_git(
        "status",
        "--porcelain",
    )

    if status is None:
        return None

    return bool(status)
