# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Watch Kalman star data during perigee passages."""

import os
from pathlib import Path


class Config:
    # Top-level data directory
    data_dir = "."

    # FITS file containing radiation data from STK
    rad_table_path = str(
        Path(os.environ["SKA"])
        / "data"
        / "stk_radiation"
        / "rad_data_2022:003:12:00:00.000-2025:365:11:59:59.000.fits"
    )

    # Start date to be used in determining IR thresholds
    ir_thresholds_start = "2023:100"

    # Stop date to be used in determining IR thresholds
    ir_thresholds_stop = "2023:200"

    # Number of cached ACA images files (~0.7 Mb each) to keep
    n_cache = 70


conf = Config()
