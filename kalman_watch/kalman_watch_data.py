# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Watch Kalman star data during perigee passages."""

import calendar
import collections
import functools
import json
import os
import re
from pathlib import Path
from typing import List, Union

import astropy.units as u
import numpy as np
from astropy.table import Table, vstack
from cheta.fetch import MSIDset
from cheta.utils import logical_intervals
from kadi.commands import get_observations
from kadi.commands.commands_v2 import get_cmds
from kadi.commands.states import get_states, reduce_states
from ska_helpers.logging import basic_logger


from dataclasses import dataclass
from chandra_aca.planets import get_earth_blocks
from cheta import fetch
import kadi.events

from cxotime import CxoTime, CxoTimeLike

from typing import TypeAlias
import scipy.signal
from chandra_aca.maude_decom import get_aca_images
from chandra_aca.transform import mag_to_count_rate
from kadi.commands import get_starcats


LOGGER = basic_logger(__name__, level="INFO")


FILE_DIR = Path(__file__).parent

# Sub-sample attitude error by 8 to reduce the number of points.
ATT_ERR_SUBSAMP = 8



class Options(collections.abc.MutableMapping):
    """
    Context manager and singleton configuration object.

    opt = Options(a=1, b=2)
    print(opt)  # {'a': 1, 'b': 2}
    opt2 = Options(a=2, c=3)
    print(opt)  # {'a': 1, 'b': 2}
    print(opt2)  # {'a': 1, 'b': 2}
    opt == opt2  # True
    with opt2:
        print(opt)  # {'a': 2, 'b': 2, 'c': 3}
    print(opt)  # {'a': 1, 'b': 2}
    with opt(a=3):
        print(opt)  # {'a': 3, 'b': 2}
    """

    _options = None

    def __init__(self, **options):
        if self.__class__._options is None:
            self.__class__._options = options
        self._new_options = options

    def __enter__(self):
        self._orig_options = self.__class__._options
        new_options = self._orig_options.copy()
        new_options.update(self._new_options)
        self.__class__._options = new_options

    def __exit__(self, type, value, traceback):
        self.__class__._options = self._orig_options

    def __call__(self, **kwargs):
        return self.__class__(**kwargs)

    def update(self, options):
        self._options.update(options)

    def __delitem__(self, key):
        self._options.__delitem__(key)
    def __getitem__(self, key):
        return self._options.__getitem__(key)
    def __iter__(self):
        return self._options.__iter__()
    def __len__(self):
        return len(self._options)
    def __setitem__(self, key, val):
        self._options[key] = val
    
    def __repr__(self):
        return self._options.__repr__()
    def __str__(self):
        return self._options.__str__()

    def perigees_dir_path(self) -> Path:
        return Path(self["data_dir"]) / self["perigee_dir_basename"]

    def perigees_index_table_path(self) -> Path:
        return self.perigees_dir_path() / self["perigee_index_basename"]

    def evt_perigee_dir_path(self, evt: "EventPerigee"):
        return self.perigees_dir_path() / evt.dirname

    def evt_perigee_data_path(self, evt: "EventPerigee") -> Path:
        return self.evt_perigee_dir_path(evt) / self["perigee_event_basename"]

    def evt_perigee_info_path(self, evt: "EventPerigee") -> Path:
        return self.evt_perigee_dir_path(evt) / self["perigee_info_basename"]

OPTIONS = Options(
    data_dir=".",
    perigee_event_basename="data.npz",
    perigee_info_basename="info.json",
    perigee_index_basename="kalman_perigees.ecsv",
    perigee_dir_basename="perigees",
    rad_table_path=str(
        Path(os.environ["SKA"])
        / "data"
        / "stk_radiation"
        / "rad_data_2022:003:12:00:00.000-2025:365:11:59:59.000.fits"
    )
)


# Default Kalman low intervals thresholds (n_kalstr, dur_limit) for
# (AOKALSTR <= n_kalstr) and (duration > dur_limit)
KALMAN_LIMITS = [
    (3, 120),
    (2, 20),
    (1, 10),
]


def get_dirname(date: Union[CxoTime, None]) -> str:
    if date is None:
        out = ""
    else:
        ymdhms = date.ymdhms
        mon = calendar.month_abbr[ymdhms.month]
        out = f"{ymdhms['year']}/{mon}-{ymdhms['day']:02d}"
    return out


def read_kalman_stats(from_info=False) -> Table:
    """Read kalman stats from file or from event info.json files.

    If ``from_info`` is True, read all individual event info files instead of
    the data file. This is also tried if the kalman stats file does not exist,
    which allows re-generating the kalman stats file.
    """
    path = OPTIONS.perigees_index_table_path()
    if path.exists() and not from_info:
        LOGGER.info(f"Reading kalman perigee data from {path}")
        kalman_stats = Table.read(path)
    else:
        rows = []
        # Look for files like 2019/Jan-12/info.json
        for info_file in OPTIONS.perigees_dir_path().glob("????/??????/info.json"):
            if re.search(r"\d{4}/\w{3}-\d{2}/info\.json", info_file.as_posix()):
                LOGGER.info(f"Reading kalman perigee data from {info_file}")
                info = json.loads(info_file.read_text())
                rows.append(info)

        LOGGER.info(f"No kalman perigee stats data found at {path}")
        LOGGER.info(f"Creating new table from {len(rows)} info files")
        kalman_stats = Table(rows=rows)
        if rows:
            kalman_stats.sort("perigee", reverse=True)

    return kalman_stats


def get_evts_perigee(
    start: CxoTime, stop: CxoTime, stats_prev: Table
) -> List["EventPerigee"]:
    """
    Get the perigee events within start/stop.

    This selects perigees within start/stop and then finds the span of
    ERs (obsid > 38000) within +/- 12 hours of perigee.

    :param start: CxoTime
        Start of date range
    :param stop: CxoTime
        End of date range
    :param stats_prev: Table
        Previous kalman stats table
    :returns: list of PerigeeEvent
        List of PerigeeEvent objects
    """
    LOGGER.info(f"Getting perigee events between {start} and {stop}")
    # event_types = ["EEF1000", "EPERIGEE", "XEF1000"]
    cmds_perigee = get_cmds(
        start=start, stop=stop, type="ORBPOINT", event_type="EPERIGEE"
    )

    # Find contiguous intervals of ERs (obsid > 38000)
    states = get_states(start - 3 * u.day, stop + 3 * u.day, state_keys=["obsid"])
    states["obsid"] = np.where(states["obsid"] > 38000, 1, 0)
    states = reduce_states(states, state_keys=["obsid"], merge_identical=True)

    dirnames_prev = stats_prev["dirname"] if len(stats_prev) > 0 else []

    events = []
    for cmd in cmds_perigee:
        t_perigee = cmd["time"]
        ok = (states["tstart"] <= t_perigee) & (t_perigee < states["tstop"])
        n_ok = np.count_nonzero(ok)
        if n_ok == 0:
            LOGGER.warning(
                "WARNING: No ER observations found covering perigee at"
                f" {CxoTime(t_perigee).date}"
            )
            continue
        elif n_ok > 1:
            raise ValueError(
                "Found multiple states covering perigee at"
                f" {CxoTime(t_perigee).date} (this really should not happen, this must"
                " be a bug"
            )

        t_rad_entry = max(states["tstart"][ok][0], t_perigee - 20000)
        t_rad_exit = min(states["tstop"][ok][0], t_perigee + 20000)

        event = EventPerigee(
            rad_entry=t_rad_entry,
            perigee=cmd["date"],
            rad_exit=t_rad_exit,
        )

        if event.dirname in dirnames_prev:
            # If the event is already in the previous kalman stats table then
            # move on silently.
            continue

        if event.tlm is not None:
            events.append(event)
        else:
            LOGGER.info(f"No TLM found for perigee event at {event.perigee}, skipping")
            continue

    LOGGER.info(f"Found {len(events)} new perigee event(s)")
    return events


def get_stats(evts_perigee) -> Table:
    """
    Get the kalman perigee stats for the given events.

    :param evts_perigee: list of PerigeeEvent
        List of PerigeeEvent objects
    :returns: Table
        Table of kalman perigee stats
    """
    rows = [evt.info for evt in reversed(evts_perigee)]

    out = Table(rows=rows)
    return out


class EventPerigee:
    """Class for tracking Kalman star data through perigee."""

    def __init__(self, rad_entry, perigee, rad_exit):
        self.rad_entry = CxoTime(rad_entry)
        self.perigee = CxoTime(perigee)
        self.rad_exit = CxoTime(rad_exit)
        self.prev_date = None
        self.next_date = None

    @property
    def prev_date(self):
        return self._prev_date

    @prev_date.setter
    def prev_date(self, value):
        self._prev_date = value if value is None else CxoTime(value)

    @property
    def next_date(self):
        return self._next_date

    @next_date.setter
    def next_date(self, value):
        self._next_date = value if value is None else CxoTime(value)

    @classmethod
    def from_npz(cls, path):
        LOGGER.info(f"Loading perigee event from {path}")
        with np.load(path) as npz_data:
            data = dict(npz_data)
        obj = cls(data["rad_entry"], data["perigee"], data["rad_exit"])

        bad = data["aokalstr"] == -1
        data["aokalstr"] = data["aokalstr"].astype(np.float64)
        data["aokalstr"][bad] = np.nan

        for slot in range(8):
            data[f"aca_track{slot}"] = (data["aca_track"] & (1 << slot)).astype(bool)
            data[f"aca_ir{slot}"] = (data["aca_ir"] & (1 << slot)).astype(bool)

        data["times"] = obj.perigee.secs + data["perigee_times"].astype(np.float64)

        obj._data = data
        obj._obss = Table(data["obss"])
        return obj

    def __repr__(self):
        return (
            f"EventPerigee(rad_entry={self.rad_entry.date!r},"
            f" perigee={self.perigee.date!r},"
            f" rad_exit={self.rad_exit.date!r})"
        )

    @property
    def dirname(self):
        return get_dirname(self.perigee)

    @property
    def obss(self):
        if not hasattr(self, "_obss"):
            LOGGER.info(f"Getting observations from kadi commands for {self.dirname}")
            obss = get_observations(start=self.rad_entry, stop=self.rad_exit)
            self._obss = Table(obss)
        return self._obss

    @property
    def tlm(self):
        if not hasattr(self, "_tlm"):
            self._tlm = self._get_tlm()
        return self._tlm

    def _get_tlm(self) -> Union[MSIDset, None]:
        """Get telemetry for processing perigee for Kalman

        :returns: MSIDset or None
        """
        msids = [
            "aokalstr",
            "aopcadmd",
            "aoacaseq",
            "aoaciir*",
            "aoacrpt",
            "aoacfct*",
            "aoatter*",
        ]
        LOGGER.info(f"Getting telemetry for {self.perigee}")
        tlm = MSIDset(msids, self.rad_entry, self.rad_exit)
        if (
            len(tlm["aokalstr"]) == 0
            or self.rad_exit.cxcsec - tlm["aokalstr"].times[-1] > 100
        ):
            return None

        tlm["aokalstr"].vals = tlm["aokalstr"].vals.astype(np.float64)

        # Reduce everything to the first ACA values during NPNT/KALM
        ok = tlm["aoacrpt"].vals.astype(int) == 0
        tlm.interpolate(
            times=tlm["aokalstr"].times[ok], bad_union=False, filter_bad=False
        )

        bad = (
            (tlm["aopcadmd"].vals != "NPNT")
            | (tlm["aoacaseq"].vals != "KALM")
            | tlm["aokalstr"].bads
        )
        tlm["aokalstr"].vals[bad] = np.nan
        for axis in range(1, 4):
            tlm[f"aoatter{axis}"].vals[bad] = np.nan

        tlm.perigee_times = tlm.times - self.perigee.cxcsec

        return tlm

    @property
    def data(self):
        """Get data for processing perigee"""
        if not hasattr(self, "_data"):
            self._data = self._get_data()
        return self._data

    def _get_data(self) -> dict:
        """Get data for processing perigee for Kalman

        :returns: dict of data
        """
        LOGGER.debug(f"Setting data property for {self.dirname}")
        data = {}
        for axis in range(1, 4):
            # Subsample by 8 since this does not vary quickly
            data[f"aoatter{axis}"] = (
                self.tlm[f"aoatter{axis}"].vals[::ATT_ERR_SUBSAMP].astype(np.float32)
            )
        data["aokalstr"] = self.tlm["aokalstr"].vals
        # fmt: off
        data["npnt_kalm"] = (
            (self.tlm["aopcadmd"].vals == "NPNT")
            & (self.tlm["aoacaseq"].vals == "KALM")
        )
        # fmt: on
        for slot in range(8):
            data[f"aca_track{slot}"] = self.tlm[f"aoacfct{slot}"].vals == "TRAK"
            data[f"aca_ir{slot}"] = self.tlm[f"aoaciir{slot}"].vals == "ERR"
        data["times"] = self.tlm["aokalstr"].times
        data["perigee_times"] = self.tlm.perigee_times.astype(np.float32)
        data["perigee"] = self.perigee.date
        data["rad_entry"] = self.rad_entry.date
        data["rad_exit"] = self.rad_exit.date
        data["obss"] = self.obss.as_array()

        return data

    @property
    def info(self):
        if not hasattr(self, "_info"):
            self._info = self._get_info()
        return self._info

    def _get_info(self):
        low_kals = self.low_kalmans
        info = {
            "dirname": self.dirname,
            "perigee": self.perigee.date[:19],
        }
        for key in ["rad_entry", "perigee", "rad_exit"]:
            info[key] = getattr(self, key).date
        for nle in (3, 2, 1):
            info[f"n{nle}_ints"] = np.count_nonzero((low_kals["n_kalstr"] == nle))
            info[f"n{nle}_cnt"] = low_kals.meta[f"n{nle}_cnt"]
        return info

    @property
    def predicted_kalman_drops(self):
        if not hasattr(self, "_info"):
            self._predicted_kalman_drops = self._get_predicted_kalman_drops()
        return self._predicted_kalman_drops

    def _get_predicted_kalman_drops(self):
        # result of a linear fit of kalman_drops vs proton_26_300_MeV
        b, a = 1.72306678e-09, 0.011492516864638577
        rad_table = get_rad_table()
        sel = (rad_table["time"] >= self.rad_entry) & (rad_table["time"] <= self.rad_exit)
        predicted_kalman_drops = {
            "times": (rad_table["time"][sel] - self.perigee).sec,
            "values": a + rad_table["proton_26_300_MeV"][sel] * b,
        }
        return predicted_kalman_drops

    def write_info(self):
        """Write info to file"""
        path = OPTIONS.evt_perigee_info_path(self)
        LOGGER.info(f"Writing info to {path}")
        path.write_text(json.dumps(self.info, indent=4))

    def write_data(self):
        # Compressed version of data
        dc = {}

        # Store aokalstr as int8 with the nan's as -1
        vals = self.data["aokalstr"].copy()
        vals[np.isnan(vals)] = -1
        dc["aokalstr"] = vals.astype(np.int8)

        for key in (
            "perigee_times",
            "npnt_kalm",
            "perigee",
            "rad_entry",
            "rad_exit",
            "obss",
            "aoatter1",
            "aoatter2",
            "aoatter3",
        ):
            dc[key] = self.data[key]

        dc["aca_track"] = np.zeros(len(self.data["times"]), dtype=np.uint8)
        dc["aca_ir"] = np.zeros(len(self.data["times"]), dtype=np.uint8)
        for slot in range(8):
            dc["aca_track"] |= self.data[f"aca_track{slot}"].astype(np.uint8) << slot
            dc["aca_ir"] |= self.data[f"aca_ir{slot}"].astype(np.uint8) << slot

        path = OPTIONS.evt_perigee_data_path(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Writing perigee data to {path}")
        np.savez_compressed(path, **dc)

    @property
    def low_kalmans(self):
        if not hasattr(self, "_low_kalmans"):
            self._low_kalmans = self._get_low_kalmans()
        return self._low_kalmans

    def _get_low_kalmans(self) -> Table:
        rows = []
        for n_kalstr, dur_limit in KALMAN_LIMITS:
            vals = self.data["aokalstr"].copy()
            vals[np.isnan(vals)] = 10
            ints_low = logical_intervals(self.data["times"], vals <= n_kalstr)
            ints_low = ints_low[ints_low["duration"] > dur_limit]

            for int_low in ints_low:
                p0 = int_low["tstart"] - self.perigee.cxcsec
                p1 = int_low["tstop"] - self.perigee.cxcsec
                row = dict(int_low)
                del row["tstart"]
                del row["tstop"]
                row["n_kalstr"] = n_kalstr
                row["tstart_rel"] = p0
                row["tstop_rel"] = p1
                rows.append(row)

        if len(rows) > 0:
            low_kalmans = Table(rows=rows)
        else:
            low_kalmans = Table(
                names=[
                    "datestart",
                    "datestop",
                    "duration",
                    "n_kalstr",
                    "tstart_rel",
                    "tstop_rel",
                ],
                dtype=[str, str, float, int, float, float],
            )
        for col in low_kalmans.itercols():
            if col.info.dtype.kind == "f":
                col.info.format = ".1f"

        for nle in (1, 2, 3):
            low_kalmans.meta[f"n{nle}_cnt"] = np.count_nonzero(vals <= nle)

        return low_kalmans

    def get_kalman_drops_nman(self):
        # having this at top-level might create a circular import
        from kalman_watch.monitor_win_perigee import (
            get_mon_dataset, get_kalman_drops_nman, NotEnoughImagesError
        )

        # Intervals of NMAN within 100 minutes of perigee
        manvrs_perigee = get_manvrs_perigee(self.rad_entry, self.rad_exit)

        # Get list of monitor window data for each perigee maneuver
        mons = []
        for manvr in manvrs_perigee:
            try:
                mon = get_mon_dataset(
                    manvr["datestart"],
                    manvr["datestop"],
                    # the following should be configured somehow
                    ir_thresholds_start="2023:100",
                    ir_thresholds_stop="2023:200",
                    data_dir="/Users/javierg/SAO/git/kalman_watch/kalman_watch3_data",
                    cache=True,
                )
                mons.append(mon)
            except NotEnoughImagesError:
                # logger.warning(err)
                pass

        if not mons:
            return {
                "times": np.array([]),
                "values": np.array([]),
            }

        kalman_drops_nman_list = [get_kalman_drops_nman(mon) for mon in mons]

        nman_table = vstack([
            table_from_perigee(kalman_drops)
            for kalman_drops in kalman_drops_nman_list
            if len(kalman_drops.times) > 0
        ])
        nman_table.sort("cxcsec")

        result = {
            "times": np.asarray(nman_table["time"]),
            "values": np.asarray(nman_table["kalman_drops"]),
        }

        return result


    def get_kalman_drops_npnt(self):

        if self.tlm is None or len(self.data["times"]) <= 200:
            return {
                "times": np.array([]),
                "values": np.array([]),
            }

        times_from_perigee, n_drops = self._get_binned_drops_from_npnt()
        result = {
            "times": times_from_perigee,
            "values": n_drops,
        }

        return result


    def _get_binned_drops_from_npnt(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the fraction of IR flags per "minute" from NPNT telemetry.

        Here a "minute" is really 60 * 1.025 seconds = 61.5, or 1.025 minutes. This
        corresponds to exactly 30 ACA image readouts (2.05 sec per image) per "minute".

        Parameters
        ----------
        ep : EventPerigee
            Perigee event object with relevant ACA telemetry from one perigee

        Returns
        -------
        time_means : np.ndarray
            Array of mean time from perigee (sec) in each bin
        ir_flag_fracs : np.ndarray
            Array of fraction of IR flags set in each bin
        """
        # Select only data in AOPCADMD in NPNT and AOACASEQ in KALM
        npnt_kalm = self.data["npnt_kalm"]
        # Time from perigee in seconds
        times_from_perigee = self.data["perigee_times"][npnt_kalm]
        if len(times_from_perigee) == 0:
            return np.array([]), np.array([])
        ir_count = np.zeros(times_from_perigee.shape, dtype=int)
        n_samp = np.zeros(times_from_perigee.shape, dtype=int)

        # Count number of IR flags set for each slot when slot is tracking
        for slot in range(8):
            ir_count[:] += np.where(
                self.data[f"aca_ir{slot}"][npnt_kalm]
                & self.data[f"aca_track{slot}"][npnt_kalm],
                1,
                0,
            )

        # Count number of slot-samples when slot is tracking
        for slot in range(8):
            n_samp[:] += np.where(
                self.data[f"aca_track{slot}"][npnt_kalm],
                1,
                0,
            )

        tbl = Table()
        tbl["idx"] = (times_from_perigee // (60 * 1.025)).astype(int)
        tbl["times_from_perigee"] = times_from_perigee
        tbl["ir_count"] = ir_count
        tbl["n_samp"] = n_samp
        tbl_grp = tbl.group_by("idx")

        time_means = []
        ir_flag_fracs = []
        for i0, i1 in zip(tbl_grp.groups.indices[:-1], tbl_grp.groups.indices[1:]):
            # For a fully-sampled "minute" there are 30 ACA telemetry samples (2.05 sec per
            # sample) times 8 slots = 240 potential samples. Require at least half of
            # samples in a "minute" in order to return a count, otherwise just no data.
            n_samp = np.sum(tbl_grp["n_samp"][i0:i1])
            if n_samp > 120:
                time_means.append(np.mean(tbl_grp["times_from_perigee"][i0:i1]))
                # Calculate fraction of available samples that have IR flag set
                ir_flag_fracs.append(np.sum(tbl_grp["ir_count"][i0:i1]) / n_samp)

        return np.array(time_means), np.array(ir_flag_fracs)


def table_from_perigee(perigee):
    perigee_time = CxoTime(perigee.perigee_date).cxcsec
    table = Table()
    table["time"] = perigee.times
    table["kalman_drops"] = perigee.kalman_drops
    table["perigee"] = perigee.perigee_date
    table["perigee_cxcsec"] = perigee_time
    # why does the following give a different (and wrong) result?
    # table["cxcsec"] = perigee_time + perigee.times
    table["cxcsec"] = table["perigee_cxcsec"] + perigee.times
    return table


#################################################################################
# These are the monitoring-window kalman drop functions
#################################################################################


@dataclass
class KalmanDropsData:
    start: CxoTime
    stop: CxoTime
    times: np.ndarray
    kalman_drops: np.ndarray
    perigee_date: str

# Typing hint for a table of images coming from chandra_aca.maude_decom.get_aca_images()
ACAImagesTable: TypeAlias = Table
MonDataSet: TypeAlias = dict[str]  # TODO: use dataclass


def get_manvrs_perigee(start: CxoTimeLike, stop: CxoTimeLike) -> Table:
    """Get maneuvers that start or stop within 100 minutes of perigee.

    This is used to select the monitor window data to compute the number of kalman drops
    per minute during perigee.

    Maneuvers are defined as intervals of AOPCADMD == "NMAN" available in the CXC cheta
    archive.

    Returns an astropy Table with a row for each maneuver interval.  Columns are:

    * datestart: date of interval start
    * datestop: date of interval stop
    * duration: duration of interval (sec)
    * tstart: time of interval start (CXC sec)
    * tstop: time of interval stop (CXC sec)

    Parameters
    ----------
    start : CxoTimeLike
        Start time
    stop : CxoTimeLike
        Stop time

    Returns
    -------
    manvrs_perigee : Table
        Table of intervals of maneuvers that are within 100 minutes of perigee.
    """
    start = CxoTime(start)
    stop = CxoTime(stop)

    LOGGER.info(f"Getting maneuvers from {start.date} to {stop.date}")
    cmds = get_cmds(start, stop)
    perigee_dates = cmds[cmds["event_type"] == "EPERIGEE"]["date"]
    pcad_mode = fetch.Msid("aopcadmd", start, stop)

    # TODO: with the "intervals" refactor WIP this could be done more nicely by taking
    # the intersection of the maneuvers and the Earth block intervals. Instead here we
    # stub in a 3rd state "EART" for Earth block intervals and report every interval
    # even though almost all of them are not during a perigee maneuver.
    LOGGER.info(f"Getting intervals of Earth blocks from {start.date} to {stop.date}")
    blocks = get_earth_blocks(start, stop)
    for block in blocks:
        LOGGER.info(
            f" Excluding Earth block from {block['datestart']} "
            f"to {block['datestop']}"
        )
        idx0, idx1 = pcad_mode.times.searchsorted([block["tstart"], block["tstop"]])
        pcad_mode.vals[idx0:idx1] = "EART"

    manvrs = logical_intervals(pcad_mode.times, pcad_mode.vals == "NMAN")

    # Get maneuvers that intersect with the interval covering -/+ 100 minutes of perigee
    manvrs_perigee = []
    for perigee_date in CxoTime(perigee_dates):
        ok = (np.abs(manvrs["tstart"] - perigee_date.secs) < 100 * 60) | (
            np.abs(manvrs["tstop"] - perigee_date.secs) < 100 * 60
        )
        manvrs_perigee.append(manvrs[ok])

    manvrs_perigee = vstack(manvrs_perigee)
    return manvrs_perigee



def get_aca_images_cached(
    start: CxoTime,
    stop: CxoTime,
    data_dir: Path,
    cache: bool = True,
) -> ACAImagesTable:
    """Get ACA images from MAUDE and cache them in a file.

    Images are cached in ``data_dir/aca_imgs_cache/``. Files outside of start/stop range
    are deleted.

    Parameters
    ----------
    start : CxoTime
        Start time
    stop : CxoTime
        Stop time
    data_dir : Path
        Directory root for cached images
    cache : bool
        If True then cache images in ``data_dir/aca_imgs_cache/``

    Returns
    -------
    imgs : ACAImagesTable
        Table of ACA image data from chandra_aca.maude_decom.get_aca_images()

    """
    LOGGER.info(f"Getting ACA images from {start.date} to {stop.date}")
    if not cache:
        return get_aca_images(start, stop)

    name = f"aca_imgs_{start.date}_{stop.date}.fits.gz"
    cache_file = data_dir / "aca_imgs_cache" / name

    if cache_file.exists():
        out = Table.read(cache_file)
    else:
        out = get_aca_images(start, stop)
        cache_file.parent.mkdir(exist_ok=True)
        out.write(cache_file)

    return out


def clean_aca_images_cache(n_cache: int, data_dir: Path):
    """Keep only the most recent ``n_cache`` files based on file creation time.

    Parameters
    ----------
    n_cache : int
        Number of cached files to keep
    data_dir : Path
        Directory root for cached images

    Returns
    -------
    None
    """
    LOGGER.info(f"Cleaning ACA images cache to keep most recent {n_cache} files")
    data_dir = Path(data_dir)
    cache_dir = data_dir / "aca_imgs_cache"

    # Keep the most recent n_cache files based on file creation time
    cache_files = sorted(
        cache_dir.glob("aca_imgs_*.fits.gz"),
        key=lambda x: x.stat().st_mtime,
    )
    for path in cache_files[:-n_cache]:
        LOGGER.info(f"Deleting {path}")
        path.unlink()


class NotEnoughImagesError(Exception):
    """Not enough images to process."""


def process_imgs(imgs: Table, slot: int) -> ACAImagesTable:
    """Process MON data images for a single slot.

    This includes calibrating, computing background by a median filter and subtracting
    background. New slots:
    * img: calibrated image (e-/s)
    * bgd: background image (e-/s) as a moving median filter of img
    * img_corr: img - bgd
    * img_sum: sum of img_corr over all pixels

    Parameters
    ----------
    imgs : Table
        Table of ACA images from chandra_aca.maude_decom.get_aca_images()
    slot : int
        Slot number (0-7)

    Returns
    -------
    imgs_slot : Table
        Table of MON data images for a single slot
    """
    # Select MON data for this slot
    imgs_slot = imgs[(imgs["IMGNUM"] == slot) & (imgs["IMGFUNC"] == 2)]
    if len(imgs_slot) < 10:
        raise NotEnoughImagesError(
            f"Not enough images for slot {slot} found {len(imgs_slot)})"
        )

    for name in imgs_slot.colnames:
        if hasattr(imgs_slot[name], "mask"):
            imgs_slot[name] = np.array(imgs_slot[name])

    # Calibrate, compute background and subtract background
    imgs_slot["img"] = imgs_slot["IMG"].data * 5.0 / 1.7  # e-/s, and get rid of mask
    imgs_slot["bgd"] = np.zeros_like(imgs_slot["img"])
    for ii in range(8):
        for jj in range(8):
            imgs_slot["bgd"][:, ii, jj] = scipy.signal.medfilt(
                imgs_slot["img"][:, ii, jj], 5
            )
    imgs_slot["img_corr"] = imgs_slot["img"] - imgs_slot["bgd"]
    imgs_slot["img_sum"] = np.sum(imgs_slot["img_corr"], axis=(1, 2))

    # Box median filter doesn't get edges right
    imgs_slot = imgs_slot[3:-3]

    return imgs_slot


def get_nearest_perigee_date(date: CxoTimeLike) -> CxoTime:
    """Get the date of the nearest perigee to ``date``.

    Parameters
    ----------
    date : CxoTimeLike
        Date

    Returns
    -------
    perigee_date : CxoTime
        Date of nearest perigee
    """
    date = CxoTime(date)
    cmds = get_cmds(date - 3 * u.d, date + 3 * u.d, event_type="EPERIGEE")
    idx = np.argmin(np.abs(cmds["time"] - date.secs))
    return CxoTime(cmds["date"][idx])


@functools.lru_cache()
def get_ir_thresholds(start: CxoTimeLike, stop: CxoTimeLike) -> np.ndarray:
    """Get IR thresholds for guide stars in the time range.

    This emulates the PEA behavior of using the maximum of 350 and star counts / 16 as
    the delta counts threshold for setting the IR flag.

    Parameters
    ----------
    start : CxoTimeLike
        Start time
    stop : CxoTimeLike
        Stop time

    Returns
    -------
    thresholds : np.ndarray
        Array of IR thresholds for each guide star in the time range
    """
    LOGGER.info(f"Getting IR thresholds from {start} to {stop}")
    obss = get_observations(start, stop)
    thresholds_list = []
    for obs in obss:
        if obs["obsid"] > 38000:
            scs = get_starcats(obsid=obs["obsid"])
            if len(scs) == 1:
                stars = scs[0]
                ok = np.isin(stars["type"], ("BOT", "GUI"))
                mags = stars["mag"][ok]
                thresholds = np.maximum(350, mag_to_count_rate(mags) / 16)
                thresholds_list.append(thresholds)

    out = np.concatenate(thresholds_list)
    return out


def get_hits(
    mon: ACAImagesTable,
    ir_thresholds_start: CxoTimeLike,
    ir_thresholds_stop: CxoTimeLike,
) -> Table:
    """Get the hits (IR flag set) for the monitor window data.

    Parameters
    ----------
    mon : ACAImagesTable
        Table of MON data images for a single slot
    ir_thresholds_start : CxoTimeLike
        Start time for sampling guide stars for IR thresholds
    ir_thresholds_stop : CxoTimeLike
        Stop time for sampling guide stars for IR thresholds

    Returns
    -------
    hits : Table
        Table of hits (IR flag set) for the monitor window data
    """
    ir_thresholds_iter = get_ir_thresholds(ir_thresholds_start, ir_thresholds_stop)

    img_sum = mon["imgs"]["img_sum"]
    img_idxs = np.where(img_sum > 500)[0]
    img_sums = img_sum[img_idxs]
    img_maxs = np.max(mon["imgs"]["img_corr"][img_idxs], axis=(1, 2))
    hit_pixels = img_sums / img_maxs

    hits = []
    for img_idx, img_sum, img_max, hit_pixel, threshold in zip(
        img_idxs, img_sums, img_maxs, hit_pixels, ir_thresholds_iter
    ):
        hit = {
            "time": mon["imgs"]["TIME"][img_idx],
            "dt_min": mon["imgs"]["dt_min"][img_idx],
            "slot": mon["imgs"]["IMGNUM"][img_idx],
            "ir_flag": img_sum > threshold,
            "img_idx": img_idx,
            "sum": img_sum,
            "max": img_max,
            "pixels": hit_pixel,
        }
        hits.append(hit)

    if hits:
        out = Table(hits)
        out["hit_idx"] = np.arange(len(hits))
    else:
        out = Table(
            names=[
                "time",
                "dt_min",
                "slot",
                "ir_flag",
                "img_idx",
                "sum",
                "max",
                "pixels",
                "hit_idx",
            ],
            dtype=[float, float, int, bool, int, float, float, float, int],
        )

    return out


def get_mon_dataset(
    start: CxoTimeLike,
    stop: CxoTimeLike,
    ir_thresholds_start: CxoTimeLike,
    ir_thresholds_stop: CxoTimeLike,
    data_dir: str | Path,
    cache: bool = True,
) -> MonDataSet:
    """Get a dataset of MON data over the time range.

    This returns a dict with keys:
    - imgs: Table of MON data images for all slots sorted by time with extra columns:
        - img_idx: index into imgs
        - dt_min: time in minutes since perigee
    - perigee_date: date of nearest perigee to the middle of the time range
    - hits: Table of hits (IR flag set) for the monitor window data
    - start: start time
    - stop: stop time

    Parameters
    ----------
    start : CxoTimeLike
        Start time
    stop : CxoTimeLike
        Stop time
    ir_thresholds_start : CxoTimeLike
        Start time for sampling guide stars for IR thresholds
    ir_thresholds_stop : CxoTimeLike
        Stop time for sampling guide stars for IR thresholds
    data_dir : str, Path
        Directory root for cached images
    cache : bool
        If True then cache images in ``data_dir/aca_imgs_cache/``

    Returns
    -------
    mon : MonDataSet
        Dataset of MON data over the time range
    """
    data_dir = Path(data_dir)
    start = CxoTime(start)
    stop = CxoTime(stop)
    start.format = "date"
    stop.format = "date"
    LOGGER.info(f"Getting MON data from {start.date} to {stop.date}")
    imgs = get_aca_images_cached(start, stop, data_dir, cache=cache)

    # Create a dataset of MON data for this slot
    mon = {}
    mons = []
    for slot in range(8):
        try:
            mons.append(process_imgs(imgs, slot))
        except NotEnoughImagesError as err:
            LOGGER.warning(err)
    if len(mons) == 0:
        raise NotEnoughImagesError(f"No image data between {start} and {stop}")
    mon["imgs"] = vstack(mons)
    mon["imgs"].sort("TIME")
    mon["imgs"]["img_idx"] = np.arange(len(mon["imgs"]))
    mon["perigee_date"] = get_nearest_perigee_date(imgs["TIME"][len(imgs) // 2])
    mon["imgs"]["dt_min"] = (mon["imgs"]["TIME"] - mon["perigee_date"].secs) / 60
    mon["start"] = start
    mon["stop"] = stop

    mon["hits"] = get_hits(mon, ir_thresholds_start, ir_thresholds_stop)

    return mon


def get_kalman_drops_per_minute(mon: MonDataSet) -> tuple[np.ndarray, np.ndarray]:
    """Get the number of drops per "minute" by counting IR flags set.

    Here a "minute" is really 60 * 1.025 seconds = 61.5, or 1.025 minutes.

    Parameters
    ----------
    mon : MonDataSet
        Dataset of MON data from get_mon_dataset()

    Returns
    -------
    dt_mins : np.ndarray
        Array of dt_min values (minutes since perigee) for each minute
    kalman_drops : np.ndarray
        Array of number of kalman drops per minute scaled
    """
    bins = np.arange(-100, 100, 1.025)
    kalman_drops = []
    dt_mins = []
    for dt_min0, dt_min1 in zip(bins[:-1], bins[1:]):
        has_imgs = (mon["imgs"]["dt_min"] >= dt_min0) & (
            mon["imgs"]["dt_min"] < dt_min1
        )
        # For a fully-sampled "minute" there are 15 ACA images per slot (4.1 sec per
        # image) times 8 slots = 120 images. Require at least half of images present
        # in order to return a count, otherwise just no data.
        if (n_imgs := np.count_nonzero(has_imgs)) > 60:
            ok = (mon["hits"]["dt_min"] >= dt_min0) & (mon["hits"]["dt_min"] < dt_min1)
            n_drops = np.sum(mon["hits"][ok]["ir_flag"]) / n_imgs
            kalman_drops.append(n_drops)
            dt_mins.append((dt_min0 + dt_min1) / 2)

    return np.array(dt_mins), np.array(kalman_drops)


def get_binned_drops_from_event_perigee(
    ep: EventPerigee,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the fraction of IR flags per "minute" from NPNT telemetry.

    Here a "minute" is really 60 * 1.025 seconds = 61.5, or 1.025 minutes. This
    corresponds to exactly 30 ACA image readouts (2.05 sec per image) per "minute".

    Parameters
    ----------
    ep : EventPerigee
        Perigee event object with relevant ACA telemetry from one perigee

    Returns
    -------
    time_means : np.ndarray
        Array of mean time from perigee (sec) in each bin
    ir_flag_fracs : np.ndarray
        Array of fraction of IR flags set in each bin
    """
    # Select only data in AOPCADMD in NPNT and AOACASEQ in KALM
    npnt_kalm = ep.data["npnt_kalm"]
    # Time from perigee in seconds
    times_from_perigee = ep.data["perigee_times"][npnt_kalm]
    if len(times_from_perigee) == 0:
        return np.array([]), np.array([])
    ir_count = np.zeros(times_from_perigee.shape, dtype=int)
    n_samp = np.zeros(times_from_perigee.shape, dtype=int)

    # Count number of IR flags set for each slot when slot is tracking
    for slot in range(8):
        ir_count[:] += np.where(
            ep.data[f"aca_ir{slot}"][npnt_kalm]
            & ep.data[f"aca_track{slot}"][npnt_kalm],
            1,
            0,
        )

    # Count number of slot-samples when slot is tracking
    for slot in range(8):
        n_samp[:] += np.where(
            ep.data[f"aca_track{slot}"][npnt_kalm],
            1,
            0,
        )

    tbl = Table()
    tbl["idx"] = (times_from_perigee // (60 * 1.025)).astype(int)
    tbl["times_from_perigee"] = times_from_perigee
    tbl["ir_count"] = ir_count
    tbl["n_samp"] = n_samp
    tbl_grp = tbl.group_by("idx")

    time_means = []
    ir_flag_fracs = []
    for i0, i1 in zip(tbl_grp.groups.indices[:-1], tbl_grp.groups.indices[1:]):
        # For a fully-sampled "minute" there are 30 ACA telemetry samples (2.05 sec per
        # sample) times 8 slots = 240 potential samples. Require at least half of
        # samples in a "minute" in order to return a count, otherwise just no data.
        n_samp = np.sum(tbl_grp["n_samp"][i0:i1])
        if n_samp > 120:
            time_means.append(np.mean(tbl_grp["times_from_perigee"][i0:i1]))
            # Calculate fraction of available samples that have IR flag set
            ir_flag_fracs.append(np.sum(tbl_grp["ir_count"][i0:i1]) / n_samp)

    return np.array(time_means), np.array(ir_flag_fracs)


def get_kalman_drops_nman(mon: MonDataSet) -> KalmanDropsData:
    """Get kalman_drops data in the peculiar form for plot_kalman_drops.

    Parameters
    ----------
    mon : MonDataSet
        Dataset of MON data from get_mon_dataset()
    idx : int
        Index of this perigee (used to assign a color)

    Returns
    -------
    kalman_drops_data : KalmanDropsData
    """
    dt_mins, kalman_drops = get_kalman_drops_per_minute(mon)
    times = np.array(dt_mins) * 60
    kalman_drops = KalmanDropsData(
        start=mon["start"],
        stop=mon["stop"],
        times=times,
        kalman_drops=kalman_drops,
        perigee_date=mon["perigee_date"].date,
    )
    return kalman_drops



def get_kalman_drops_npnt(start, stop, duration=100) -> list[KalmanDropsData]:
    """Get the fraction of IR flags set per minute from NPNT telemetry.

    Parameters
    ----------
    start : CxoTimeLike
        Start time
    stop : CxoTimeLike
        Stop time
    duration : int
        Duration around perigee in minutes (default=100)

    Returns
    -------
    kalman_drops_data : list[KalmanDropsData]
    """
    start = CxoTime(start)
    stop = CxoTime(stop)
    rad_zones = kadi.events.rad_zones.filter(start, stop).table
    perigee_times = CxoTime(rad_zones["perigee"])
    event_perigees = get_perigee_events(perigee_times, duration)

    kalman_drops_list = []
    for ep in event_perigees:
        if ep.tlm is None:
            LOGGER.warning(f"No telemetry for perigee {ep.perigee.date}")
            continue
        if len(ep.data["times"]) > 200:
            times_from_perigee, n_drops = get_binned_drops_from_event_perigee(ep)
            kalman_drops = KalmanDropsData(
                start=start,
                stop=stop,
                times=times_from_perigee,
                kalman_drops=n_drops,
                perigee_date=ep.perigee.date,
            )
            kalman_drops_list.append(kalman_drops)

    return kalman_drops_list


@functools.lru_cache()
def get_rad_table():
    rad_table = Table.read(OPTIONS["rad_table_path"])
    rad_table["time"] = CxoTime(rad_table["time"])
    return rad_table


def get_kalman_drops_prediction(start, stop, duration=100) -> list[KalmanDropsData]:
    """Get the expected fraction of IR flags set per minute from from STK radiation model.

    Parameters
    ----------
    start : CxoTimeLike
        Start time
    stop : CxoTimeLike
        Stop time
    duration : int
        Duration around perigee in minutes (default=100)

    Returns
    -------
    kalman_drops_data : list[KalmanDropsData]
    """
    start = CxoTime(start)
    stop = CxoTime(stop)
    rad_zones = kadi.events.rad_zones.filter(start, stop).table
    perigee_times = CxoTime(rad_zones["perigee"])
    event_perigees = get_perigee_events(perigee_times, duration)

    result = [
        KalmanDropsData(
            start=perigee.rad_entry,
            stop=perigee.rad_exit,
            times=perigee.predicted_kalman_drops["times"],
            kalman_drops=perigee.predicted_kalman_drops["values"],
            perigee_date=perigee.perigee,
        )
        for perigee in event_perigees
    ]
    return result


def get_perigee_events(perigee_times: np.ndarray, duration=100) -> list[EventPerigee]:
    """Get perigee times

    Parameters
    ----------
    perigee_times : np.ndarray
        Array of perigee times
    duration : int
        Duration around perigee in minutes (default=100)

    Returns
    -------
    event_perigees : list[EventPerigee]
        List of perigee events
    """
    event_perigees = []
    for perigee_time in perigee_times:
        event_perigee = EventPerigee(
            perigee_time - duration * u.min,
            perigee_time,
            perigee_time + duration * u.min,
        )
        event_perigees.append(event_perigee)
    return event_perigees


def _reshape_to_n_sample_2d(arr: np.ndarray, n_sample: int = 60) -> np.ndarray:
    """Reshape 1D array to 2D with one row per n_sample samples."""
    arr = arr[: -(arr.size % n_sample)]
    arr = arr.reshape(-1, n_sample)
    return arr

