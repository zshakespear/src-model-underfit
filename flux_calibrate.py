"""
The purpose of this script is to flux-calibrate a normalized spectrum.

The script takes three commandline arguments:
    - Address of the spectrum in Jy (the script will still work with any F_nu unit)
    - Address to the config json which lists the filters in SVO and magnitudes to use for calibration
    - Address to where to write the normalized spectrum
    
Note that the config json must have errors for each spectrum and those errors
must appear in the same order that the filters do.
"""

import seda
import argparse
import json
import numpy as np

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_add", help="Address of the spectrum in Jy")
    parser.add_argument("config_json", help="Path to JSON file with filters and magnitudes.")
    parser.add_argument(
        "out_add",
        help="Path to write the calibrated spectrum to",
    )
    
    return parser.parse_args()

def load_config(path: str) -> dict:
    """Load and validate the configuration JSON.

    Parameters
    ----------
    path : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Configuration mapping.
    """
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)


    return config

def phot_error(scale : float, spec : np.ndarray, config : dict):
    """ Error function for minimization algorithm.
    
    Parameters
    ----------
    scale : float
        Multiplicative factor for spectrum
        
    spec : np.ndarray
        Numpy grid containing spectrum in Jy or F_nu. Assumes that the flux
        is located in column 1
        
    config : dict
        Dictionary containing the list of filters and mangitudes
    
    Returns
    -------
    float
        L2 norm of the error between synthetic and observed photometry
    """
    scaled_flux = scale * spec[:,1]
    obs = list()
    pred = list()
    
    for k in config.keys():
        obs.append(config[k])
        seda_res = seda.synthetic_photometry.flux_to_mag(scaled_flux, k)
        pred.append(seda_res['mag'][0])
        
    pred = np.array(pred)
    obs = np.array(obs)
    
    return np.linalg.norm(pred - obs, 2)

def main():
    args = parse_args()
    in_grid = np.loadtxt(args.object_add)
    config = load_config(args.config_json)
    
    filters = list()
    mags = list()
    mag_errs = list()
    
    for k in config.keys():
        if k.find('err') != -1:
            mag_errs.append(config[k])
        else:
            filters.append(k)
            mags.append(config[k])
    seda_res = seda.synthetic_photometry.calibrate_spectrum(
        wl = in_grid[:,0],
        flux = in_grid[:,1],
        eflux = in_grid[:,2],
        flux_unit = "Jy",
        filters = filters,
        mag = mags,
        emag = mag_errs
        )
    out_grid = np.array([
        seda_res['wl'],
        seda_res['flux'],
        seda_res['eflux']
        ]).T
    
    np.savetxt(args.out_add, out_grid, header = 'Wavelength (micron), Flux (Jy), eFlux (Jy)')

if __name__ == "__main__":
    main()