#!/usr/bin/env python3
"""Combine pre-calibrated SpeX, AKARI, and Spitzer spectra for one object.

Note: this script was written with the assistance of Perplexity AI.

This script expects two command-line arguments:
1. An object identifier string.
2. A path to a JSON configuration file containing the keys
   ``spex_path``, ``akari_path``, ``spitzer_path``, and ``out_path``.

All input spectra are assumed to be whitespace-delimited files with three
columns:
- wavelength [micron]
- flux [Jy]
- flux_error [Jy]

Workflow
--------
1. Find exactly one matching file for each instrument.
2. Load each spectrum and clip to the 1--15 micron range.
3. Apply a Hampel filter to each instrument separately.
4. Resolve overlap by keeping the highest-S/N instrument over overlapping
   wavelength intervals, where S/N is taken to be |flux| / error.
5. Write the combined result to NetCDF.

Example
-------
python combine_hampel_spectra.py J1234+5678 config.json

Note: this code was written with the assistance of Perplexity AI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from flameskimmer_tools import get_current_author, get_run_timestamp_iso


MIN_WAVELENGTH_UM = 1.0
MAX_WAVELENGTH_UM = 15.0


class SpectrumLoadError(RuntimeError):
    """Raised when a spectrum file cannot be loaded or validated."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_id", help="Object identifier used to match filenames.")
    parser.add_argument("config_json", help="Path to JSON file with required directory paths.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=11,
        help="Odd Hampel window size in samples (default: 11).",
    )
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=3.0,
        help="Hampel threshold in robust sigma units (default: 3.0).",
    )
    parser.add_argument(
        "--min-error-floor-frac",
        type=float,
        default=1e-3,
        help="Minimum local scale floor as a fraction of the local median absolute flux (default: 1e-3).",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    """Load and validate the configuration JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = ["spex_path", "akari_path", "spitzer_path", "out_path"]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing configuration keys: {missing}")
    return config


def find_unique_file(base_path: str | Path, object_id: str, allowed_suffixes: Tuple[str, ...]) -> Path:
    """Find exactly one matching file under a directory tree."""
    base = Path(base_path).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Input path does not exist: {base}")

    object_id_lower = object_id.lower()
    matches = [
        p for p in base.rglob("*")
        if p.is_file()
        and object_id_lower in p.name.lower()
        and p.suffix.lower() in allowed_suffixes
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No matching file for '{object_id}' under {base} with suffixes {allowed_suffixes}"
        )
    if len(matches) > 1:
        joined = "\n  ".join(str(p) for p in matches)
        raise RuntimeError(
            f"Expected exactly one file for '{object_id}' under {base}, found {len(matches)}:\n  {joined}"
        )
    return matches[0]


def _read_numeric_text_file(path: Path, expected_ncols: int = 3) -> np.ndarray:
    """Read a whitespace-delimited numeric text file."""
    try:
        data = np.genfromtxt(path, comments="#", dtype=float)
    except Exception as exc:
        raise SpectrumLoadError(f"Could not read text data from {path}") from exc

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.ndim != 2 or data.shape[1] != expected_ncols:
        raise SpectrumLoadError(
            f"Expected {expected_ncols} columns in {path}, found shape {data.shape}"
        )

    finite_rows = np.all(np.isfinite(data), axis=1)
    data = data[finite_rows]
    if data.size == 0:
        raise SpectrumLoadError(f"No finite numeric rows found in {path}")
    return data


def load_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a three-column spectrum in microns and Jy, clipped to 1--15 micron."""
    data = _read_numeric_text_file(path, expected_ncols=3)
    wavelength_um = data[:, 0]
    flux_jy = data[:, 1]
    error_jy = data[:, 2]

    mask = (
        np.isfinite(wavelength_um)
        & np.isfinite(flux_jy)
        & np.isfinite(error_jy)
        & (wavelength_um >= MIN_WAVELENGTH_UM)
        & (wavelength_um <= MAX_WAVELENGTH_UM)
        & (error_jy > 0)
    )
    wavelength_um = wavelength_um[mask]
    flux_jy = flux_jy[mask]
    error_jy = error_jy[mask]

    if wavelength_um.size == 0:
        raise SpectrumLoadError(
            f"No valid data remain in {path} after clipping to {MIN_WAVELENGTH_UM}--{MAX_WAVELENGTH_UM} micron"
        )

    order = np.argsort(wavelength_um)
    return wavelength_um[order], flux_jy[order], error_jy[order]


def hampel_filter(
    values: np.ndarray,
    window_size: int = 11,
    n_sigma: float = 3.0,
    min_error_floor_frac: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a defensive Hampel filter to a one-dimensional array."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    x = np.asarray(values, dtype=float)
    y = x.copy()
    outlier = np.zeros_like(x, dtype=bool)
    local_scale = np.full_like(x, np.nan, dtype=float)
    half_window = window_size // 2
    mad_scale = 1.4826

    for i in range(len(x)):
        left = max(0, i - half_window)
        right = min(len(x), i + half_window + 1)
        window = x[left:right]
        window = window[np.isfinite(window)]
        if window.size < 3 or not np.isfinite(x[i]):
            continue

        median = np.median(window)
        mad = np.median(np.abs(window - median))
        abs_median = np.median(np.abs(window))
        floor = max(np.finfo(float).eps, min_error_floor_frac * max(abs_median, np.abs(median), 1.0))
        sigma = max(mad_scale * mad, floor)
        local_scale[i] = sigma

        if np.abs(x[i] - median) > n_sigma * sigma:
            y[i] = median
            outlier[i] = True

    return y, outlier, local_scale


def filter_instrument(
    source_name: str,
    wavelength_um: np.ndarray,
    flux_jy: np.ndarray,
    error_jy: np.ndarray,
    window_size: int,
    n_sigma: float,
    min_error_floor_frac: float,
) -> Dict[str, np.ndarray]:
    """Apply Hampel filtering to one instrument spectrum."""
    flux_clean_jy, outlier_mask, local_scale = hampel_filter(
        flux_jy,
        window_size=window_size,
        n_sigma=n_sigma,
        min_error_floor_frac=min_error_floor_frac,
    )
    snr = np.abs(flux_clean_jy) / error_jy
    return {
        "source": np.full(wavelength_um.size, source_name, dtype=object),
        "wavelength_um": wavelength_um,
        "flux_jy": flux_jy,
        "flux_clean_jy": flux_clean_jy,
        "flux_error_jy": error_jy,
        "snr": snr,
        "is_hampel_outlier": outlier_mask,
        "hampel_local_scale": local_scale,
    }


def _coverage_bounds(wavelength_um: np.ndarray) -> Tuple[float, float]:
    """Return the wavelength coverage bounds for one spectrum."""
    return float(np.min(wavelength_um)), float(np.max(wavelength_um))


def _median_snr(block: Dict[str, np.ndarray], interval_min: float, interval_max: float) -> float:
    """Compute the median S/N of a block inside a wavelength interval."""
    mask = (block["wavelength_um"] >= interval_min) & (block["wavelength_um"] <= interval_max)
    if not np.any(mask):
        return -np.inf
    return float(np.nanmedian(block["snr"][mask]))


def resolve_overlaps_by_interval(filtered_blocks: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Resolve overlaps using interval-based segment selection with highest S/N.

    The wavelength axis is partitioned by all instrument coverage boundaries. In
    each resulting interval, the instrument with the highest median S/N over that
    interval is selected, and only that instrument contributes points there.
    """
    bounds = []
    for block in filtered_blocks:
        lo, hi = _coverage_bounds(block["wavelength_um"])
        bounds.extend([lo, hi])
    bounds = np.array(sorted(set(bounds)), dtype=float)

    kept_segments = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        if right <= left:
            continue
        active = []
        for block in filtered_blocks:
            lo, hi = _coverage_bounds(block["wavelength_um"])
            if hi > left and lo < right:
                active.append(block)
        if not active:
            continue

        best_block = max(active, key=lambda b: _median_snr(b, left, right))
        mask = (best_block["wavelength_um"] >= left) & (best_block["wavelength_um"] < right)
        if right == bounds[-1]:
            mask = (best_block["wavelength_um"] >= left) & (best_block["wavelength_um"] <= right)
        if np.any(mask):
            kept_segments.append({key: best_block[key][mask] for key in best_block})

    if not kept_segments:
        raise SpectrumLoadError("No spectrum segments remained after overlap resolution")

    combined = {key: np.concatenate([seg[key] for seg in kept_segments]) for key in kept_segments[0]}
    order = np.argsort(combined["wavelength_um"])
    return {key: combined[key][order] for key in combined}


def build_dataset(combined: Dict[str, np.ndarray]) -> xr.Dataset:
    """Build an xarray Dataset from the combined spectrum."""
    npts = combined["wavelength_um"].size
    return xr.Dataset(
        data_vars={
            "flux_jy": ("point", combined["flux_jy"]),
            "flux_clean_jy": ("point", combined["flux_clean_jy"]),
            "flux_error_jy": ("point", combined["flux_error_jy"]),
            "snr": ("point", combined["snr"]),
            "is_hampel_outlier": ("point", combined["is_hampel_outlier"]),
            "hampel_local_scale": ("point", combined["hampel_local_scale"]),
            "source": ("point", combined["source"].astype(str)),
        },
        coords={
            "point": np.arange(npts),
            "wavelength_um": ("point", combined["wavelength_um"]),
        },
        attrs={
            "flux_units": "Jy",
            "wavelength_units": "um",
            "wavelength_range_um": f"{MIN_WAVELENGTH_UM} to {MAX_WAVELENGTH_UM}",
            "cleaning_method": "Per-instrument Hampel filter using local median and MAD with scale floor",
            "overlap_rule": "interval-based segment selection using highest median S/N",
        },
    )


def main() -> None:
    """Run the full command-line workflow."""
    args = parse_args()
    config = load_config(args.config_json)

    spex_file = find_unique_file(config["spex_path"], args.object_id, allowed_suffixes=(".txt", ".dat"))
    akari_file = find_unique_file(config["akari_path"], args.object_id, allowed_suffixes=(".txt", ".dat"))
    spitzer_file = find_unique_file(config["spitzer_path"], args.object_id, allowed_suffixes=(".txt", ".dat"))

    spex_wav, spex_flux, spex_err = load_spectrum(spex_file)
    akari_wav, akari_flux, akari_err = load_spectrum(akari_file)
    spitzer_wav, spitzer_flux, spitzer_err = load_spectrum(spitzer_file)

    filtered_blocks = [
        filter_instrument(
            "spex", spex_wav, spex_flux, spex_err,
            window_size=args.window_size,
            n_sigma=args.n_sigma,
            min_error_floor_frac=args.min_error_floor_frac,
        ),
        filter_instrument(
            "akari", akari_wav, akari_flux, akari_err,
            window_size=args.window_size,
            n_sigma=args.n_sigma,
            min_error_floor_frac=args.min_error_floor_frac,
        ),
        filter_instrument(
            "spitzer", spitzer_wav, spitzer_flux, spitzer_err,
            window_size=args.window_size,
            n_sigma=args.n_sigma,
            min_error_floor_frac=args.min_error_floor_frac,
        ),
    ]

    combined = resolve_overlaps_by_interval(filtered_blocks)
    ds = build_dataset(combined)
    ds.attrs["object_id"] = args.object_id
    ds.attrs["input_spex_file"] = str(spex_file)
    ds.attrs["input_akari_file"] = str(akari_file)
    ds.attrs["input_spitzer_file"] = str(spitzer_file)
    ds.attrs["hampel_window_size"] = args.window_size
    ds.attrs["hampel_n_sigma"] = args.n_sigma
    ds.attrs["hampel_min_error_floor_frac"] = args.min_error_floor_frac
    ds.attrs["author"] = get_current_author()
    ds.attrs["date_created"] = get_run_timestamp_iso()

    out_dir = Path(config["out_path"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.object_id}_combined.nc"
    ds.to_netcdf(out_file)
    print(out_file)


if __name__ == "__main__":
    main()