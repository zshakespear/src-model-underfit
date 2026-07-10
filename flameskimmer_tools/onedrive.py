from pathlib import Path
from typing import Iterable
import subprocess
import time

def iter_files(root: Path, pattern: str) -> Iterable[Path]:
    """Yield matching files under a root directory.

    Parameters
    ----------
    root : Path
        Root directory to scan.
    pattern : str
        Glob-style pattern.

    Yields
    ------
    Path
        Matching file path.
    """
    for path in root.rglob(pattern):
        if path.is_file():
            yield path


def run_attrib(path: Path, *flags: str) -> None:
    """Run Windows attrib on a path.

    Parameters
    ----------
    path : Path
        File path to modify.
    *flags : str
        attrib flags such as '+U', '-P', '-U', '+P'.
    """
    subprocess.run(["attrib", *flags, str(path)], check=False, shell=True)


def hydrate_file(path: Path, wait_seconds: float = 2.0, retries: int = 30) -> None:
    """Ensure a OneDrive file is available locally.

    Parameters
    ----------
    path : Path
        File to hydrate.
    wait_seconds : float
        Delay between retries.
    retries : int
        Number of hydration checks.

    Raises
    ------
    RuntimeError
        If hydration fails.
    """
    run_attrib(path, "-U", "+P")
    for _ in range(retries):
        try:
            with path.open("rb") as handle:
                handle.read(1)
            return
        except OSError:
            time.sleep(wait_seconds)
    raise RuntimeError(f"Could not hydrate file: {path}")


def dehydrate_file(path: Path) -> None:
    """Return a file to online-only state.

    Parameters
    ----------
    path : Path
        File to dehydrate.
    """
    run_attrib(path, "+U", "-P")

# FIXME: docstrings
def output_path_for_source(source_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_path = source_path.relative_to(input_root)
    return (output_root / relative_path).with_suffix(".nc")


def already_regridded(source_path: Path, input_root: Path, output_root: Path) -> bool:
    output_path = output_path_for_source(source_path, input_root, output_root)
    return output_path.is_file()