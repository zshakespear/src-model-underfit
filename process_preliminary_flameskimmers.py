"""
Strip preliminary FlameSkimmer models one at a time from a command line entry point.

Example
-------
python process_preliminary_flameskimmers.py "C:\\Users\\you\\OneDrive\\BigFolder" --pattern "*.nc"

Notes
-----
This script is intended for Windows + OneDrive Files On-Demand.
It avoids reading file contents until the moment a file is processed.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Iterable
import xarray as xr
from tqdm import tqdm
import numpy as np


def existing_dir(value: str) -> Path:
    """
    Validate and return a directory path.

    Parameters
    ----------
    value : str
        Directory path from the command line.

    Returns
    -------
    Path
        Resolved directory path.

    Raises
    ------
    argparse.ArgumentTypeError
        If the path does not exist or is not a directory.
    """
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {value}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path is not a directory: {value}")
    return path


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process OneDrive files one at a time without keeping them all local."
    )
    parser.add_argument(
        "root",
        type=existing_dir,
        help="Path to the large OneDrive folder.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob-style filename pattern, e.g. '*.fits' or '*.csv'. Default: '*'",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Do not return files to online-only after processing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without hydrating or modifying anything.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to process.",
    )
    parser.add_argument(
    "--txt-dir",
    type=existing_or_new_dir,
    default=None,
    help="Directory where .txt outputs should be saved. "
         "If omitted, save next to each .nc file.",
)
    return parser.parse_args()


def iter_files(root: Path, pattern: str) -> Iterable[Path]:
    """
    Yield matching files under a root directory.

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
    """
    Run Windows attrib on a path.

    Parameters
    ----------
    path : Path
        File path to modify.
    *flags : str
        attrib flags such as '+U', '-P', '-U', '+P'.
    """
    subprocess.run(
        ["attrib", *flags, str(path)],
        check=False,
        shell=True,
    )


def hydrate_file(path: Path, wait_seconds: float = 2.0, retries: int = 30) -> None:
    """
    Ensure a OneDrive file is available locally.

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
    # "Always available" style flags for download/hydration.
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
    """
    Return a file to online-only state.

    Parameters
    ----------
    path : Path
        File to dehydrate.
    """
    run_attrib(path, "+U", "-P")
    
def existing_or_new_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Not a directory: {value}")
    return path

def process_file(path: Path, root: Path, txt_dir: Path | None) -> None:
    """
    Perform the per-file work.

    Replace this function with your real processing logic.

    Parameters
    ----------
    path : Path
        Local hydrated file.
    """
    
    if txt_dir is None:
        txt_path = path.with_suffix(".txt")
    else:
        relative_path = path.relative_to(root)
        txt_path = (txt_dir / relative_path).with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)

    txt_path.write_text("your output here\n")

    
    full_model = xr.open_dataset(path)
    wavs = full_model['wavelength'].to_numpy()
    flux = full_model['flux_emission'].to_numpy()
    stripped_model = np.stack([wavs, flux], axis=0)
    # FIXME: Header metadata is hardcoded
    np.savetxt(txt_path, stripped_model.T, delimiter=',', header='# Wavelength (microns), Flux (erg/cm**2/s/cm\n# Created by Zac Shakespear using process_preliminary_flameskimmers.py on 2026-06-24 from the .nc file of the same name in the preliminary flameskimmer grid.')
    dehydrate_file(txt_path)
    

def main() -> None:
    """
    Run the command-line workflow.
    """
    args = parse_args()

    for path in tqdm(iter_files(args.root, args.pattern)):
        if args.dry_run:
            continue

        try:
            hydrate_file(path)
            process_file(path, args.root, args.txt_dir)
        except Exception as exc:
            print(f"ERROR: {path} -> {exc}")
        finally:
            if not args.keep_local and not args.dry_run:
                dehydrate_file(path)




if __name__ == "__main__":
    main()