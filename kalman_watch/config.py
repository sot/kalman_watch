# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Watch Kalman star data during perigee passages."""

import os
from pathlib import Path

import astropy.config as _config_


class ConfigNamespace(_config_.ConfigNamespace):
    rootname = 'kalman_watch'

    data_dir=_config_.ConfigItem(
        ".",
        "Top-level data directory",
    )
    perigee_event_basename=_config_.ConfigItem(
        "data.npz",
        "Numpy compressed file name where to store perigee event data",
    )
    perigee_info_basename=_config_.ConfigItem(
        "info.json",
        "JSON file name where to store perigee event data",
    )
    perigee_index_basename=_config_.ConfigItem(
        "kalman_perigees.ecsv",
        "ECSV file name where to store perigee event data",
    )
    perigee_dir_basename=_config_.ConfigItem(
        "perigees",
        "descr",
    )
    rad_table_path=_config_.ConfigItem(str(
        Path(os.environ["SKA"])
        / "data"
        / "stk_radiation"
        / "rad_data_2022:003:12:00:00.000-2025:365:11:59:59.000.fits"
    ),
        "Fits file containing radiation data from STK",
    )
    ir_thresholds_start=_config_.ConfigItem(
        "2023:100",
        "Start date to be used in determining IR thresholds",
    )
    ir_thresholds_stop=_config_.ConfigItem(
        "2023:200",
        "Stop date to be used in determining IR thresholds",
    )

    def __str__(self):
        from pprint import pformat
        return pformat({k: getattr(self, k) for k in self})


conf = ConfigNamespace()
