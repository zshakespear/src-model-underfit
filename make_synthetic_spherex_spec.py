"""Construct synthetic SPHEREx spectra from a large NetCDF model library.

This command-line tool is designed for Windows + OneDrive Files-On-Demand.
It hydrates each source model only when needed, reads a library of SPHEREx
spectral response functions (SRFs), integrates each model spectrum through
those SRFs, writes a compact NetCDF output, and optionally returns local
files to cloud-only state.

Example
-------
python make_synthetic_spherex_spec.py \
    "D:\\spherex_srfs" \
    "C:\\Users\\you\\OneDrive\\FlameSkimmerModels" \
    "D:\\synthetic_spherex"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from flameskimmer_tools import (
    dehydrate_file,
    existing_dir,
    existing_or_new_dir,
    extract_1d_spectrum,
    find_wavelength_variable,
    get_current_author,
    get_run_timestamp_iso,
    get_wavelength_unit,
    hydrate_file,
    iter_files,
    validate_wavelength_array,
    WavelengthUnitError
)


DEFAULT_FLUX_VARIABLE = "flux_emission"
DEFAULT_SRF_RESPONSE_CANDIDATES = ("response", "throughput", "srf", "transmission")


@dataclass(frozen=True)
class SpectralResponseFunction:
    """Container for one SPHEREx spectral response function.

    Parameters
    ----------
    channel_id : str
        Identifier for the SRF channel.
    wavelength : np.ndarray
        One-dimensional wavelength array.
    response : np.ndarray
        One-dimensional response array.
    wavelength_unit : str
        Wavelength unit string.
    source_file : Path
        File from which the SRF was loaded.
    """

    channel_id: str
    wavelength: np.ndarray
    response: np.ndarray
    wavelength_unit: str
    source_file: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Construct synthetic SPHEREx spectra by integrating NetCDF model "
            "spectra through a directory of SRFs."
        )
    )
    parser.add_argument(
        "srf_dir",
        type=existing_dir,
        help="Directory containing SPHEREx spectral response function files.",
    )
    parser.add_argument(
        "model_root",
        type=existing_dir,
        help="Path to the large OneDrive folder containing source model .nc files.",
    )
    parser.add_argument(
        "output_dir",
        type=existing_or_new_dir,
        help="Directory where synthetic SPHEREx NetCDF outputs should be saved.",
    )
    parser.add_argument(
        "--pattern",
        default="*.nc",
        help="Glob-style filename pattern for source model files. Default: '*.nc'",
    )
    parser.add_argument(
        "--srf-pattern",
        default="*.nc",
        help="Glob-style filename pattern for SRF files. Default: '*.nc'",
    )
    parser.add_argument(
        "--flux-variable",
        default=DEFAULT_FLUX_VARIABLE,
        help=f"Name of the flux variable to convolve. Default: '{DEFAULT_FLUX_VARIABLE}'",
    )
    parser.add_argument(
        "--wavelength-variable",
        default=None,
        help=(
            "Name of the model wavelength variable. If omitted, the script uses "
            "the shared flameskimmer_tools wavelength-discovery helper."
        ),
    )
    parser.add_argument(
        "--srf-wavelength-variable",
        default=None,
        help=(
            "Name of the wavelength variable in the SRF files. If omitted, the "
            "script uses the shared flameskimmer_tools wavelength-discovery helper."
        ),
    )
    parser.add_argument(
        "--srf-response-variable",
        default=None,
        help=(
            "Name of the response variable in the SRF files. If omitted, the "
            "script tries common candidates such as response, throughput, srf, "
            "and transmission."
        ),
    )
    parser.add_argument(
        "--channel-id-attr",
        default="channel_id",
        help="SRF attribute name to use as the channel identifier. Default: 'channel_id'",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Do not return processed model files to online-only after processing.",
    )
    parser.add_argument(
        "--keep-srf-local",
        action="store_true",
        help="Do not return SRF files to online-only after loading them.",
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
        help="Maximum number of model files to process.",
    )
    return parser.parse_args()


def find_response_variable(dataset: xr.Dataset, preferred_name: str | None) -> str:
    """Find the SRF response variable name in a dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to inspect.
    preferred_name : str or None
        Explicit response variable name supplied by the user.

    Returns
    -------
    str
        Name of the response variable.

    Raises
    ------
    KeyError
        If no suitable response variable can be found.
    """
    if preferred_name is not None:
        if preferred_name not in dataset:
            raise KeyError(f"SRF response variable '{preferred_name}' not found.")
        return preferred_name

    for name in DEFAULT_SRF_RESPONSE_CANDIDATES:
        if name in dataset:
            return name

    for name in dataset.variables:
        lower = name.lower()
        if any(token in lower for token in ("response", "throughput", "trans", "srf")):
            return name

    raise KeyError("Could not determine SRF response variable name.")


def validate_response_array(response: np.ndarray, label: str) -> np.ndarray:
    """Validate and standardize an SRF response array.

    Parameters
    ----------
    response : np.ndarray
        Input response array.
    label : str
        Human-readable label for error messages.

    Returns
    -------
    np.ndarray
        One-dimensional response array.

    Raises
    ------
    ValueError
        If the array is not one-dimensional, finite, or contains only zeros.
    """
    response = np.asarray(response, dtype=float)
    if response.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if not np.all(np.isfinite(response)):
        raise ValueError(f"{label} contains non-finite values.")
    if np.any(response < 0.0):
        raise ValueError(f"{label} contains negative values.")
    if np.all(response == 0.0):
        raise ValueError(f"{label} is zero everywhere.")
    return response


def load_single_srf(
    srf_path: Path,
    wavelength_variable: str | None,
    response_variable: str | None,
    channel_id_attr: str,
) -> SpectralResponseFunction:
    """Load one SRF from a NetCDF file.

    Parameters
    ----------
    srf_path : Path
        Path to the SRF file.
    wavelength_variable : str or None
        Explicit wavelength variable name if supplied.
    response_variable : str or None
        Explicit response variable name if supplied.
    channel_id_attr : str
        Attribute name to use for the channel identifier.

    Returns
    -------
    SpectralResponseFunction
        Parsed and validated SRF.
    """
    with xr.open_dataset(srf_path) as dataset:
        wavelength_name = find_wavelength_variable(dataset, wavelength_variable)
        response_name = find_response_variable(dataset, response_variable)
        wavelength_unit = get_wavelength_unit(dataset, wavelength_name, str(srf_path))
        wavelength = validate_wavelength_array(
            dataset[wavelength_name].to_numpy(),
            f"SRF wavelength '{wavelength_name}' in {srf_path.name}",
        )
        response = validate_response_array(
            dataset[response_name].to_numpy(),
            f"SRF response '{response_name}' in {srf_path.name}",
        )
        if response.shape[0] != wavelength.shape[0]:
            raise ValueError(
                f"SRF response '{response_name}' length does not match wavelength length "
                f"in {srf_path.name}."
            )

        channel_id = (
            dataset.attrs.get(channel_id_attr)
            or dataset[response_name].attrs.get(channel_id_attr)
            or srf_path.stem
        )

    return SpectralResponseFunction(
        channel_id=str(channel_id),
        wavelength=np.asarray(wavelength, dtype=float),
        response=np.asarray(response, dtype=float),
        wavelength_unit=str(wavelength_unit),
        source_file=srf_path,
    )


def load_srf_library(
    srf_dir: Path,
    pattern: str,
    wavelength_variable: str | None,
    response_variable: str | None,
    channel_id_attr: str,
    keep_srf_local: bool,
) -> list[SpectralResponseFunction]:
    """Load and validate a directory of SPHEREx SRFs.

    Parameters
    ----------
    srf_dir : Path
        Directory containing SRF files.
    pattern : str
        Glob-style pattern for SRF files.
    wavelength_variable : str or None
        Explicit SRF wavelength variable name if supplied.
    response_variable : str or None
        Explicit SRF response variable name if supplied.
    channel_id_attr : str
        Attribute name to use for the channel identifier.
    keep_srf_local : bool
        Whether to keep SRF files local after loading.

    Returns
    -------
    list[SpectralResponseFunction]
        Loaded SRFs sorted by channel wavelength center.
    """
    srfs: list[SpectralResponseFunction] = []

    for srf_path in iter_files(srf_dir, pattern):
        hydrate_file(srf_path)
        try:
            srfs.append(
                load_single_srf(
                    srf_path=srf_path,
                    wavelength_variable=wavelength_variable,
                    response_variable=response_variable,
                    channel_id_attr=channel_id_attr,
                )
            )
        finally:
            if not keep_srf_local:
                dehydrate_file(srf_path)

    if not srfs:
        raise ValueError(f"No SRF files found in {srf_dir} matching pattern '{pattern}'.")

    first_unit = srfs[0].wavelength_unit
    for srf in srfs[1:]:
        if srf.wavelength_unit != first_unit:
            raise WavelengthUnitError(
                "SRF wavelength units do not match across the library: "
                f"'{first_unit}' vs '{srf.wavelength_unit}'."
            )

    srfs.sort(key=lambda srf: float(np.average(srf.wavelength, weights=srf.response)))
    return srfs


def validate_model_and_srf_units(model_unit: str, srf_unit: str, model_path: Path) -> None:
    """Validate that model and SRF wavelength units match exactly.

    Parameters
    ----------
    model_unit : str
        Model wavelength unit string.
    srf_unit : str
        SRF wavelength unit string.
    model_path : Path
        Model file used for error messages.

    Raises
    ------
    WavelengthUnitError
        If the units do not match.
    """
    if model_unit != srf_unit:
        raise WavelengthUnitError(
            f"Model wavelength units '{model_unit}' in {model_path} do not match "
            f"SRF wavelength units '{srf_unit}'."
        )


def integrate_spectrum_through_srf(
    model_wavelength: np.ndarray,
    model_flux: np.ndarray,
    srf: SpectralResponseFunction,
) -> float:
    """Integrate one model spectrum through one SRF.

    Parameters
    ----------
    model_wavelength : np.ndarray
        Model wavelength array.
    model_flux : np.ndarray
        Model flux array.
    srf : SpectralResponseFunction
        Spectral response function.

    Returns
    -------
    float
        Bandpass-weighted synthetic flux.

    Raises
    ------
    ValueError
        If the model does not cover the SRF support or if normalization fails.
    """
    srf_mask = srf.response > 0.0
    srf_wavelength = srf.wavelength[srf_mask]
    srf_response = srf.response[srf_mask]

    if srf_wavelength.size < 2:
        raise ValueError(f"SRF '{srf.channel_id}' does not have enough nonzero support.")

    model_min = float(model_wavelength[0])
    model_max = float(model_wavelength[-1])
    srf_min = float(srf_wavelength[0])
    srf_max = float(srf_wavelength[-1])

    if srf_min < model_min or srf_max > model_max:
        raise ValueError(
            f"Model wavelength coverage [{model_min}, {model_max}] does not fully span "
            f"SRF '{srf.channel_id}' support [{srf_min}, {srf_max}]."
        )

    interpolated_flux = np.interp(srf_wavelength, model_wavelength, model_flux)
    numerator = np.trapezoid(interpolated_flux * srf_response, x=srf_wavelength)
    denominator = np.trapezoid(srf_response, x=srf_wavelength)

    if denominator == 0.0:
        raise ValueError(f"SRF '{srf.channel_id}' normalization integral is zero.")

    return float(numerator / denominator)


def make_synthetic_spherex_spectrum(
    model_wavelength: np.ndarray,
    model_flux: np.ndarray,
    srfs: list[SpectralResponseFunction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a synthetic SPHEREx spectrum for one model.

    Parameters
    ----------
    model_wavelength : np.ndarray
        Model wavelength array.
    model_flux : np.ndarray
        Model flux array.
    srfs : list[SpectralResponseFunction]
        Spectral response functions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Channel effective wavelengths, synthetic fluxes, and channel IDs.
    """
    channel_wavelengths: list[float] = []
    synthetic_fluxes: list[float] = []
    channel_ids: list[str] = []

    for srf in srfs:
        flux_value = integrate_spectrum_through_srf(model_wavelength, model_flux, srf)
        effective_wavelength = float(
            np.trapezoid(srf.wavelength * srf.response, x=srf.wavelength)
            / np.trapezoid(srf.response, x=srf.wavelength)
        )
        channel_wavelengths.append(effective_wavelength)
        synthetic_fluxes.append(flux_value)
        channel_ids.append(srf.channel_id)

    return (
        np.asarray(channel_wavelengths, dtype=float),
        np.asarray(synthetic_fluxes, dtype=float),
        np.asarray(channel_ids, dtype="U"),
    )


def output_path_for_source(source_path: Path, root: Path, output_dir: Path) -> Path:
    """Construct the output NetCDF path for a source model.

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


def write_synthetic_spherex_output(
    output_path: Path,
    channel_wavelengths: np.ndarray,
    wavelength_unit: str,
    synthetic_fluxes: np.ndarray,
    flux_variable: str,
    channel_ids: np.ndarray,
    source_model_path: Path,
    srf_dir: Path,
) -> None:
    """Write one synthetic SPHEREx spectrum to NetCDF.

    Parameters
    ----------
    output_path : Path
        Destination NetCDF path.
    channel_wavelengths : np.ndarray
        Effective channel wavelengths.
    wavelength_unit : str
        Wavelength unit string.
    synthetic_fluxes : np.ndarray
        Synthetic SPHEREx fluxes.
    flux_variable : str
        Name of the source flux variable.
    channel_ids : np.ndarray
        Channel identifiers.
    source_model_path : Path
        Path to the original model file.
    srf_dir : Path
        Directory containing the SRF library.
    """
    author = get_current_author()
    run_timestamp = get_run_timestamp_iso()

    dataset = xr.Dataset(
        data_vars={
            "synthetic_flux": (
                ("channel",),
                synthetic_fluxes,
                {"source_flux_variable": flux_variable},
            ),
            "channel_id": (("channel",), channel_ids),
        },
        coords={
            "channel": np.arange(channel_wavelengths.size, dtype=int),
            "wavelength": (
                ("channel",),
                channel_wavelengths,
                {"units": wavelength_unit},
            ),
        },
        attrs={
            "source_model_file": str(source_model_path),
            "srf_directory": str(srf_dir),
            "script_name": Path(__file__).name,
            "author": author,
            "date_created": run_timestamp,
            "processing_history": (
                f"Synthetic SPHEREx spectrum generated with {Path(__file__).name} "
                f"by {author} on {run_timestamp} using SRFs from {srf_dir}."
            ),
        },
    )

    encoding = {
        "wavelength": {"dtype": "float64"},
        "synthetic_flux": {"dtype": "float64", "zlib": True, "complevel": 4},
    }
    dataset.to_netcdf(output_path, encoding=encoding)
    dataset.close()


def process_model_file(
    model_path: Path,
    model_root: Path,
    output_dir: Path,
    srfs: list[SpectralResponseFunction],
    flux_variable: str,
    preferred_wavelength_name: str | None,
    srf_dir: Path,
) -> Path:
    """Process one model into a synthetic SPHEREx spectrum.

    Parameters
    ----------
    model_path : Path
        Local hydrated model file.
    model_root : Path
        Root of the model tree.
    output_dir : Path
        Root of the output tree.
    srfs : list[SpectralResponseFunction]
        Loaded SRF library.
    flux_variable : str
        Name of the flux variable to convolve.
    preferred_wavelength_name : str or None
        Explicit wavelength variable name if supplied.
    srf_dir : Path
        Directory containing the SRF library.

    Returns
    -------
    Path
        Output file path written for this source model.
    """
    output_path = output_path_for_source(model_path, model_root, output_dir)

    with xr.open_dataset(model_path) as dataset:
        wavelength_name = find_wavelength_variable(dataset, preferred_wavelength_name)
        model_unit = get_wavelength_unit(dataset, wavelength_name, str(model_path))
        validate_model_and_srf_units(model_unit, srfs[0].wavelength_unit, model_path)
        model_wavelength, model_flux = extract_1d_spectrum(
            dataset,
            wavelength_name,
            flux_variable,
        )

    channel_wavelengths, synthetic_fluxes, channel_ids = make_synthetic_spherex_spectrum(
        model_wavelength,
        model_flux,
        srfs,
    )

    write_synthetic_spherex_output(
        output_path=output_path,
        channel_wavelengths=channel_wavelengths,
        wavelength_unit=srfs[0].wavelength_unit,
        synthetic_fluxes=synthetic_fluxes,
        flux_variable=flux_variable,
        channel_ids=channel_ids,
        source_model_path=model_path,
        srf_dir=srf_dir,
    )
    return output_path


def main() -> None:
    """Run the command-line workflow."""
    args = parse_args()

    srfs = load_srf_library(
        srf_dir=args.srf_dir,
        pattern=args.srf_pattern,
        wavelength_variable=args.srf_wavelength_variable,
        response_variable=args.srf_response_variable,
        channel_id_attr=args.channel_id_attr,
        keep_srf_local=args.keep_srf_local,
    )

    processed = 0
    for model_path in tqdm(iter_files(args.model_root, args.pattern)):
        if args.limit is not None and processed >= args.limit:
            break

        if args.dry_run:
            print(model_path)
            processed += 1
            continue

        try:
            hydrate_file(model_path)
            output_path = process_model_file(
                model_path=model_path,
                model_root=args.model_root,
                output_dir=args.output_dir,
                srfs=srfs,
                flux_variable=args.flux_variable,
                preferred_wavelength_name=args.wavelength_variable,
                srf_dir=args.srf_dir,
            )
            if not args.keep_output_local:
                dehydrate_file(output_path)
        except Exception as exc:
            print(f"ERROR: {model_path} -> {exc}")
        finally:
            if not args.keep_local:
                dehydrate_file(model_path)

        processed += 1


if __name__ == "__main__":
    main()