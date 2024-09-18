from pathlib import Path

from kalman_watch import conf


def data_dir() -> Path:
    return Path(conf.data_dir)


def perigees_dir_path() -> Path:
    return data_dir() / conf.perigee_dir_basename


def perigees_index_table_path() -> Path:
    return perigees_dir_path() / conf.perigee_index_basename
