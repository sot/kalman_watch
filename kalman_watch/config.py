# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Watch Kalman star data during perigee passages."""

import os
from pathlib import Path


class Config:
    # Top-level data directory
    data_dir = Path(".")

    # FITS file containing radiation data from STK
    rad_table_path = (
        Path(os.environ["SKA"])
        / "data"
        / "stk"
        / "radiation_data.fits"
    )

    # Start date to be used in determining IR thresholds
    ir_thresholds_start = "2023:100"

    # Stop date to be used in determining IR thresholds
    ir_thresholds_stop = "2023:200"

    # Number of cached ACA images files (~0.7 Mb each) to keep
    n_cache = 70


conf = Config()
