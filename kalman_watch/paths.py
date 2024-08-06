from pathlib import Path

from kalman_watch import OPTIONS


def data_dir() -> Path:
    return Path(OPTIONS.data_dir)


def perigees_dir_path() -> Path:
    return data_dir() / OPTIONS.perigee_dir_basename


def perigees_index_table_path() -> Path:
    return perigees_dir_path() / OPTIONS.perigee_index_basename
