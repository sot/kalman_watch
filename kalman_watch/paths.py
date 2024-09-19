from pathlib import Path

from kalman_watch import conf


def data_dir() -> Path:
    return Path(conf.data_dir)


def perigees_dir_path() -> Path:
    return data_dir() / "perigees"


def perigees_index_table_path() -> Path:
    return perigees_dir_path() / "kalman_perigees.ecsv"


def rad_table_path() -> Path:
    return Path(conf.rad_table_path)
