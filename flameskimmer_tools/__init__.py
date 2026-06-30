"""Shared utilities for FlameSkimmer model processing workflows.

This package provides reusable helpers for command-line interfaces,
OneDrive Files-On-Demand handling, NetCDF wavelength metadata,
spectral-array validation, and provenance metadata.

Examples
--------
Import helpers directly from the package:

    from flameskimmer_tools import hydrate_file, extract_1d_spectrum

Or import submodules explicitly:

    from flameskimmer_tools import onedrive, spectra
"""

from .cli import existing_dir, existing_file, existing_or_new_dir
from .metadata import get_current_author, get_run_timestamp_iso
from .netcdf_utils import WavelengthUnitError, find_wavelength_variable, get_wavelength_unit
from .onedrive import dehydrate_file, hydrate_file, iter_files, run_attrib
from .spectra import extract_1d_spectrum, validate_wavelength_array

__all__ = [
    "existing_dir",
    "existing_file",
    "existing_or_new_dir",
    "get_current_author",
    "get_run_timestamp_iso",
    "find_wavelength_variable",
    "get_wavelength_unit",
    "dehydrate_file",
    "hydrate_file",
    "iter_files",
    "run_attrib",
    "extract_1d_spectrum",
    "validate_wavelength_array",
    "WavelengthUnitError"
]