"""Regrid preliminary FlameSkimmer NetCDF models one at a time.

This command-line tool is designed for Windows + OneDrive Files On-Demand.
It hydrates each source NetCDF file only when needed, validates wavelength
metadata against a target wavelength-grid NetCDF file, regrids the requested
flux variable onto the target wavelength grid with a cubic spline, writes a
compact NetCDF output, and optionally returns local files to cloud-only state.

Example
-------
python regrid_preliminary_flameskimmers_regridded.py \
    "C:\\Users\\you\\OneDrive\\BigFolder" \
    --wavelength-grid "D:\\reference\\target_wavelength_grid.nc" \
    --output-dir "D:\\rebinned_flameskimmers"
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
from tqdm import tqdm


DEFAULT_WAVELENGTH_CANDIDATES = ("wavelength", "wavel", "lambda", "lam")
DEFAULT_FLUX_VARIABLE = "flux_emission"


class WavelengthUnitError(ValueError):
    """Raised when wavelength units are missing or inconsistent."""


def existing_dir(value: str) -> Path:
    """Validate and return an existing directory path.

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


def existing_file(value: str) -> Path:
    """Validate and return an existing file path.

    Parameters
    ----------
    value : str
        File path from the command line.

    Returns
    -------
    Path
        Resolved file path.

    Raises
    ------
    argparse.ArgumentTypeError
        If the path does not exist or is not a file.
    """
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {value}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path is not a file: {value}")
    return path


def existing_or_new_dir(value: str) -> Path:
    """Create a directory if needed and return its resolved path.

    Parameters
    ----------
    value : str
        Directory path from the command line.

    Returns
    -------
    Path
        Resolved directory path.
    """
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Not a directory: {value}")
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Regrid preliminary FlameSkimmer NetCDF files one at a time "
            "onto a target wavelength grid."
        )
    )
    parser.add_argument(
        "root",
        type=existing_dir,
        help="Path to the large OneDrive folder containing source .nc files.",
    )
    parser.add_argument(
        "--wavelength-grid",
        required=True,
        type=existing_file,
        help="Path to the NetCDF file containing the target wavelength grid.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=existing_or_new_dir,
        help="Directory where rebinned NetCDF outputs should be saved.",
    )
    parser.add_argument(
        "--pattern",
        default="*.nc",
        help="Glob-style filename pattern for source files. Default: '*.nc'",
    )
    parser.add_argument(
        "--flux-variable",
        default=DEFAULT_FLUX_VARIABLE,
        help=f"Name of the flux variable to regrid. Default: '{DEFAULT_FLUX_VARIABLE}'",
    )
    parser.add_argument(
        "--wavelength-variable",
        default=None,
        help=(
            "Name of the wavelength variable. If omitted, the script tries common "
            "candidates such as wavelength, wavel, lambda, and lam."
        ),
    )
    parser.add_argument(
        "--bc-type",
        default="natural",
        choices=("natural", "not-a-knot", "clamped"),
        help=(
            "Boundary condition for scipy.interpolate.CubicSpline. "
            "Default: natural"
        ),
    )
    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Allow spline extrapolation outside the source wavelength range.",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Do not return processed files to online-only after processing.",
    )
    parser.add_argument(
        "--keep-output-local",
        action="store_true",
        help="Do not return output files to online-only after writing them.",
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
    return parser.parse_args()


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


def find_wavelength_variable(dataset: xr.Dataset, preferred_name: str | None) -> str:
    """Find the wavelength variable name in a dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to inspect.
    preferred_name : str or None
        Explicit variable name supplied by the user.

    Returns
    -------
    str
        Name of the wavelength variable.

    Raises
    ------
    KeyError
        If no suitable wavelength variable can be found.
    """
    if preferred_name is not None:
        if preferred_name not in dataset:
            raise KeyError(f"Wavelength variable '{preferred_name}' not found.")
        return preferred_name

    for name in DEFAULT_WAVELENGTH_CANDIDATES:
        if name in dataset:
            return name
        if name in dataset.coords:
            return name

    for name in dataset.variables:
        lower = name.lower()
        if "wave" in lower or lower in {"lambda", "lam"}:
            return name

    raise KeyError("Could not determine wavelength variable name.")


def get_wavelength_unit(dataset: xr.Dataset, variable_name: str, dataset_label: str) -> str:
    """Return the wavelength unit string for a dataset variable.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the wavelength variable.
    variable_name : str
        Wavelength variable name.
    dataset_label : str
        Human-readable label for error messages.

    Returns
    -------
    str
        Unit string.

    Raises
    ------
    WavelengthUnitError
        If the wavelength unit attribute is missing or empty.
    """
    variable = dataset[variable_name]
    for key in ("units", "unit"):
        value = variable.attrs.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise WavelengthUnitError(
        f"Wavelength units are missing for variable '{variable_name}' in {dataset_label}."
    )


def validate_matching_wavelength_units(
    source_dataset: xr.Dataset,
    source_wavelength_name: str,
    target_dataset: xr.Dataset,
    target_wavelength_name: str,
    source_label: str,
    target_label: str,
) -> str:
    """Validate that source and target wavelength units are defined and match.

    Parameters
    ----------
    source_dataset : xr.Dataset
        Source dataset.
    source_wavelength_name : str
        Source wavelength variable name.
    target_dataset : xr.Dataset
        Target grid dataset.
    target_wavelength_name : str
        Target wavelength variable name.
    source_label : str
        Label used in error messages for the source dataset.
    target_label : str
        Label used in error messages for the target dataset.

    Returns
    -------
    str
        The shared unit string.

    Raises
    ------
    WavelengthUnitError
        If units are missing or do not match exactly.
    """
    source_unit = get_wavelength_unit(source_dataset, source_wavelength_name, source_label)
    target_unit = get_wavelength_unit(target_dataset, target_wavelength_name, target_label)

    if source_unit != target_unit:
        raise WavelengthUnitError(
            "Source and target wavelength units do not match: "
            f"'{source_unit}' in {source_label} vs '{target_unit}' in {target_label}."
        )

    return source_unit


def validate_wavelength_array(wavelength: np.ndarray, label: str) -> np.ndarray:
    """Validate and standardize a wavelength array for spline interpolation.

    Parameters
    ----------
    wavelength : np.ndarray
        Input wavelength array.
    label : str
        Human-readable label for error messages.

    Returns
    -------
    np.ndarray
        One-dimensional wavelength array.

    Raises
    ------
    ValueError
        If the array is not one-dimensional, finite, or strictly increasing.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    if wavelength.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if wavelength.size < 2:
        raise ValueError(f"{label} must contain at least two points.")
    if not np.all(np.isfinite(wavelength)):
        raise ValueError(f"{label} contains non-finite values.")
    if not np.all(np.diff(wavelength) > 0.0):
        raise ValueError(f"{label} must be strictly increasing for spline interpolation.")
    return wavelength


def prepare_source_arrays(
    dataset: xr.Dataset,
    wavelength_name: str,
    flux_variable: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and validate source wavelength and flux arrays.

    Parameters
    ----------
    dataset : xr.Dataset
        Source dataset.
    wavelength_name : str
        Name of the wavelength variable.
    flux_variable : str
        Name of the flux variable.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated wavelength and flux arrays.

    Raises
    ------
    KeyError
        If the flux variable is not present.
    ValueError
        If dimensions or values are unsuitable for interpolation.
    """
    if flux_variable not in dataset:
        raise KeyError(f"Flux variable '{flux_variable}' not found.")

    wavelength = np.asarray(dataset[wavelength_name].to_numpy(), dtype=float)
    flux = np.asarray(dataset[flux_variable].to_numpy(), dtype=float)

    wavelength = validate_wavelength_array(wavelength, f"Source wavelength '{wavelength_name}'")

    if flux.ndim != 1:
        raise ValueError(f"Flux variable '{flux_variable}' must be one-dimensional.")
    if flux.shape[0] != wavelength.shape[0]:
        raise ValueError(
            f"Flux variable '{flux_variable}' length does not match wavelength length."
        )
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"Flux variable '{flux_variable}' contains non-finite values.")

    return wavelength, flux


def regrid_flux(
    source_wavelength: np.ndarray,
    source_flux: np.ndarray,
    target_wavelength: np.ndarray,
    bc_type: str,
    allow_extrapolation: bool,
) -> np.ndarray:
    """Regrid a flux array onto the target wavelength grid with a cubic spline.

    Parameters
    ----------
    source_wavelength : np.ndarray
        Source wavelength array.
    source_flux : np.ndarray
        Source flux array.
    target_wavelength : np.ndarray
        Target wavelength array.
    bc_type : str
        Boundary condition name for ``scipy.interpolate.CubicSpline``.
    allow_extrapolation : bool
        Whether to evaluate outside the source wavelength domain.

    Returns
    -------
    np.ndarray
        Rebinned flux values on the target wavelength grid.
    """
    spline = CubicSpline(
        source_wavelength,
        source_flux,
        bc_type=bc_type,
        extrapolate=allow_extrapolation,
    )
    rebinned = spline(target_wavelength)
    return np.asarray(rebinned, dtype=float)


def output_path_for_source(source_path: Path, root: Path, output_dir: Path) -> Path:
    """Construct the output NetCDF path for a source file.

    Parameters
    ----------
    source_path : Path
        Path to the source file.
    root : Path
        Root of the source tree.
    output_dir : Path
        Root of the output tree.

    Returns
    -------
    Path
        Output file path with preserved relative structure.
    """
    relative_path = source_path.relative_to(root)
    output_path = (output_dir / relative_path).with_suffix(".nc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_regridded_output(
    output_path: Path,
    target_wavelength_name: str,
    target_wavelength: np.ndarray,
    wavelength_unit: str,
    flux_variable: str,
    rebinned_flux: np.ndarray,
    source_path: Path,
    source_flux_attrs: dict,
) -> None:
    """Write the rebinned spectrum to a compact NetCDF file.

    Parameters
    ----------
    output_path : Path
        Destination NetCDF path.
    target_wavelength_name : str
        Name to use for the wavelength coordinate.
    target_wavelength : np.ndarray
        Target wavelength grid.
    wavelength_unit : str
        Wavelength unit string.
    flux_variable : str
        Name of the flux variable.
    rebinned_flux : np.ndarray
        Rebinned flux values.
    source_path : Path
        Original source file path.
    source_flux_attrs : dict
        Attributes copied from the source flux variable.
    """
    dataset = xr.Dataset(
        data_vars={
            flux_variable: (
                (target_wavelength_name,),
                rebinned_flux,
                dict(source_flux_attrs),
            )
        },
        coords={
            target_wavelength_name: (
                (target_wavelength_name,),
                target_wavelength,
                {"units": wavelength_unit},
            )
        },
        attrs={
            "source_file": str(source_path),
            "processing_history": (
                "Regridded with scipy.interpolate.CubicSpline in "
                "process_preliminary_flameskimmers_regridded.py"
            ),
        },
    )

    encoding = {
        target_wavelength_name: {"dtype": "float64"},
        flux_variable: {"dtype": "float64", "zlib": True, "complevel": 4},
    }
    dataset.to_netcdf(output_path, encoding=encoding)
    dataset.close()


def process_file(
    path: Path,
    root: Path,
    output_dir: Path,
    target_wavelength: np.ndarray,
    target_wavelength_name: str,
    target_wavelength_unit: str,
    flux_variable: str,
    preferred_wavelength_name: str | None,
    bc_type: str,
    allow_extrapolation: bool,
) -> Path:
    """Process one hydrated source file.

    Parameters
    ----------
    path : Path
        Local hydrated source NetCDF file.
    root : Path
        Root of the input tree.
    output_dir : Path
        Root of the output tree.
    target_wavelength : np.ndarray
        Target wavelength grid.
    target_wavelength_name : str
        Name of the target wavelength coordinate.
    target_wavelength_unit : str
        Shared wavelength unit string.
    flux_variable : str
        Name of the flux variable to regrid.
    preferred_wavelength_name : str or None
        Explicit wavelength variable name if supplied.
    bc_type : str
        Boundary condition for the spline.
    allow_extrapolation : bool
        Whether extrapolation is allowed.

    Returns
    -------
    Path
        Output path written for this source file.
    """
    output_path = output_path_for_source(path, root, output_dir)

    with xr.open_dataset(path) as source_dataset:
        source_wavelength_name = find_wavelength_variable(source_dataset, preferred_wavelength_name)
        validate_matching_wavelength_units(
            source_dataset,
            source_wavelength_name,
            xr.Dataset(
                coords={
                    target_wavelength_name: (
                        (target_wavelength_name,),
                        target_wavelength,
                        {"units": target_wavelength_unit},
                    )
                }
            ),
            target_wavelength_name,
            str(path),
            "target wavelength grid",
        )
        source_wavelength, source_flux = prepare_source_arrays(
            source_dataset,
            source_wavelength_name,
            flux_variable,
        )
        source_flux_attrs = dict(source_dataset[flux_variable].attrs)

    rebinned_flux = regrid_flux(
        source_wavelength,
        source_flux,
        target_wavelength,
        bc_type=bc_type,
        allow_extrapolation=allow_extrapolation,
    )

    write_regridded_output(
        output_path=output_path,
        target_wavelength_name=target_wavelength_name,
        target_wavelength=target_wavelength,
        wavelength_unit=target_wavelength_unit,
        flux_variable=flux_variable,
        rebinned_flux=rebinned_flux,
        source_path=path,
        source_flux_attrs=source_flux_attrs,
    )
    return output_path


def load_target_grid(
    wavelength_grid_path: Path,
    preferred_wavelength_name: str | None,
) -> tuple[np.ndarray, str, str]:
    """Load and validate the target wavelength grid once at startup.

    Parameters
    ----------
    wavelength_grid_path : Path
        NetCDF file containing the target wavelength grid.
    preferred_wavelength_name : str or None
        Explicit wavelength variable name if supplied.

    Returns
    -------
    tuple[np.ndarray, str, str]
        Target wavelength array, wavelength variable name, and unit string.
    """
    with xr.open_dataset(wavelength_grid_path) as dataset:
        wavelength_name = find_wavelength_variable(dataset, preferred_wavelength_name)
        wavelength_unit = get_wavelength_unit(dataset, wavelength_name, str(wavelength_grid_path))
        wavelength = validate_wavelength_array(
            dataset[wavelength_name].to_numpy(),
            f"Target wavelength '{wavelength_name}'",
        )
    return wavelength, wavelength_name, wavelength_unit


def main() -> None:
    """Run the command-line workflow."""
    args = parse_args()

    hydrate_file(args.wavelength_grid)
    try:
        target_wavelength, target_wavelength_name, target_wavelength_unit = load_target_grid(
            args.wavelength_grid,
            args.wavelength_variable,
        )
    finally:
        if not args.keep_local:
            dehydrate_file(args.wavelength_grid)

    processed = 0
    for path in tqdm(iter_files(args.root, args.pattern)):
        if args.limit is not None and processed >= args.limit:
            break

        if args.dry_run:
            print(path)
            processed += 1
            continue

        try:
            hydrate_file(path)
            output_path = process_file(
                path=path,
                root=args.root,
                output_dir=args.output_dir,
                target_wavelength=target_wavelength,
                target_wavelength_name=target_wavelength_name,
                target_wavelength_unit=target_wavelength_unit,
                flux_variable=args.flux_variable,
                preferred_wavelength_name=args.wavelength_variable,
                bc_type=args.bc_type,
                allow_extrapolation=args.allow_extrapolation,
            )
            if not args.keep_output_local:
                dehydrate_file(output_path)
        except Exception as exc:
            print(f"ERROR: {path} -> {exc}")
        finally:
            if not args.keep_local:
                dehydrate_file(path)

        processed += 1


if __name__ == "__main__":
    main()