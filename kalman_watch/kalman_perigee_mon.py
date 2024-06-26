# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Watch Kalman star data during perigee passages."""

import argparse
import calendar
import json
import re
from itertools import cycle
from pathlib import Path
from typing import List, Union

import astropy.units as u
import numpy as np
import plotly.express as px
import plotly.graph_objects as pgo
from acdc.common import send_mail
from astropy.table import Table, vstack
from cheta.fetch import MSIDset
from cheta.utils import logical_intervals
from cxotime import CxoTime
from jinja2 import Template
from kadi.commands import get_observations
from kadi.commands.commands_v2 import get_cmds
from kadi.commands.states import get_states, reduce_states
from plotly.subplots import make_subplots
from ska_helpers.logging import basic_logger
from ska_helpers.run_info import log_run_info

from kalman_watch import __version__

LOGGER = basic_logger(__name__, level="INFO")


FILE_DIR = Path(__file__).parent

# Sub-sample attitude error by 8 to reduce the number of points.
ATT_ERR_SUBSAMP = 8


def PERIGEES_DIR_PATH(data_dir: str) -> Path:
    return Path(data_dir) / "perigees"


def PERIGEES_INDEX_TABLE_PATH(data_dir: str) -> Path:
    return PERIGEES_DIR_PATH(data_dir) / "kalman_perigees.ecsv"


def EVT_PERIGEE_DATA_PATH(data_dir: str, evt: "EventPerigee") -> Path:
    return EVT_PERIGEE_DIR_PATH(data_dir, evt) / "data.npz"


def EVT_PERIGEE_INFO_PATH(data_dir: str, evt: "EventPerigee") -> Path:
    return EVT_PERIGEE_DIR_PATH(data_dir, evt) / "info.json"


def EVT_PERIGEE_DIR_PATH(data_dir: str, evt: "EventPerigee"):
    return PERIGEES_DIR_PATH(data_dir) / evt.dirname


def INDEX_DETAIL_PATH():
    return FILE_DIR / "index_kalman_perigee_detail.html"


def INDEX_LIST_PATH():
    return FILE_DIR / "index_kalman_perigee_list.html"


# Default Kalman low intervals thresholds (n_kalstr, dur_limit) for
# (AOKALSTR <= n_kalstr) and (duration > dur_limit)
KALMAN_LIMITS = [
    (3, 120),
    (2, 20),
    (1, 10),
]

# Colors for AOKALSTR plot taken from plotly default color cycle. Removed ones
# that are too close to red which is used for low kalman intervals.
COLOR_CYCLE = [
    "#1f77b4",  # muted blue
    #    '#ff7f0e',  # safety orange
    "#2ca02c",  # cooked asparagus green
    #     '#d62728',  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    #     '#e377c2',  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


def get_dirname(date: Union[CxoTime, None]) -> str:
    if date is None:
        out = ""
    else:
        ymdhms = date.ymdhms
        mon = calendar.month_abbr[ymdhms.month]
        out = f"{ymdhms['year']}/{mon}-{ymdhms['day']:02d}"
    return out


def get_opt() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kalman star watch {}".format(__version__)
    )
    parser.add_argument("--stop", type=str, help="Stop date (default=NOW)")
    parser.add_argument(
        "--lookback", type=float, default=14, help="Lookback time (days, default=14)"
    )
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    parser.add_argument(
        "--email",
        action="append",
        dest="emails",
        default=[],
        help="Email address for notification",
    )
    parser.add_argument(
        "--make-html", action="store_true", help="Make static HTML pages"
    )
    return parser


def main(sys_args=None):
    opt: argparse.Namespace = get_opt().parse_args(sys_args)
    log_run_info(LOGGER.info, opt, version=__version__)

    stop = CxoTime(opt.stop)
    start: CxoTime = stop - opt.lookback * u.day

    stats_prev = read_kalman_stats(opt)
    evts_perigee = get_evts_perigee(start, stop, stats_prev)

    # Bail out if there are no new perigee events. Since we backed up to the
    # previous perigee we need at least two perigee events.
    if len(evts_perigee) == 0:
        LOGGER.info("No new perigee events, exiting")
        return

    for evt_perigee in evts_perigee:
        evt_perigee.write_data(opt)
        evt_perigee.write_info(opt)

    stats_new = get_stats(evts_perigee)
    # Combine new and previous statistics summary
    stats_all = vstack([stats_new, stats_prev])
    stats_all.sort("perigee", reverse=True)

    path = PERIGEES_INDEX_TABLE_PATH(opt.data_dir)
    LOGGER.info(f"Writing perigee data {path}")
    stats_all.write(path, format="ascii.ecsv", overwrite=True)

    # Check for any low kalman intervals in the new events.
    has_low_kalmans = any(
        len(evt_perigee.low_kalmans) > 0 for evt_perigee in evts_perigee
    )
    if opt.emails and has_low_kalmans:
        send_process_mail(opt, evts_perigee)

    if opt.make_html:
        # Data collection and alerts are done. Now generate the web pages. This
        # is mostly for testing. In production this is done dynamically using
        # kadi web apps.
        for date_next, stat, date_prev in zip(
            [None] + stats_all["perigee"][:-1].tolist(),
            stats_all,
            stats_all["perigee"][1:].tolist() + [None],
        ):
            if stat["perigee"] < start.date:
                break
            evt_tmp = EventPerigee(stat["rad_entry"], stat["perigee"], stat["rad_exit"])
            path = EVT_PERIGEE_DATA_PATH(opt.data_dir, evt_tmp)
            evt = EventPerigee.from_npz(path)
            evt.prev_date = date_prev
            evt.next_date = date_next
            evt.make_detail_page(opt)

        make_index_list_pages(opt, stats_all)


def read_kalman_stats(opt, from_info=False) -> Table:
    """Read kalman stats from file or from event info.json files.

    If ``from_info`` is True, read all individual event info files instead of
    the data file. This is also tried if the kalman stats file does not exist,
    which allows re-generating the kalman stats file.
    """
    path = PERIGEES_INDEX_TABLE_PATH(opt.data_dir)
    if path.exists() and not from_info:
        LOGGER.info(f"Reading kalman perigee data from {path}")
        kalman_stats = Table.read(path)
    else:
        rows = []
        # Look for files like 2019/Jan-12/info.json
        for info_file in PERIGEES_DIR_PATH(opt.data_dir).glob("????/??????/info.json"):
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


def make_index_list_pages(opt, stats_all: Table) -> None:
    html = get_index_html_recent(stats_all)
    path = PERIGEES_DIR_PATH(opt.data_dir) / "index.html"
    LOGGER.info(f"Writing recent index list page to {path}")
    path.write_text(html)

    # Write index page for the last two calendar years
    years_all = CxoTime(stats_all["perigee"]).ymdhms.year
    years_unique = sorted(np.unique(years_all))
    for year in years_unique[-2:]:
        html = get_index_html_year(stats_all, year)
        path = PERIGEES_DIR_PATH(opt.data_dir) / str(year) / "index.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Writing index list page for {year} to {path}")
        path.write_text(html)


def get_index_html_year(stats_all, year):
    years_all = CxoTime(stats_all["perigee"]).ymdhms.year
    years_unique = sorted(np.unique(years_all))
    template = Template(INDEX_LIST_PATH().read_text())
    ok = years_all == year
    stats_year = stats_all[ok]
    # Strip out the year/ from dirname since we are already in the year/ dir
    stats_year["dirname"] = [dirname[5:] for dirname in stats_year["dirname"]]
    prev = f"../{year - 1}" if (year - 1) in years_unique else None
    index = "../"
    next = f"../{year + 1}" if (year + 1) in years_unique else None
    description = f"{year}"
    html = get_index_list_page(
        template,
        stats_year,
        prev=prev,
        index=index,
        next=next,
        description=description,
    )
    return html


def get_index_html_recent(stats_all):
    years_all = CxoTime(stats_all["perigee"]).ymdhms.year

    template = Template(INDEX_LIST_PATH().read_text())

    # Write index page for last 90 days of perigee data
    ok = CxoTime(stats_all["perigee"][0]) - CxoTime(stats_all["perigee"]) < 90 * u.day
    stats_recent = stats_all[ok]
    description = "last 90 days"
    html = get_index_list_page(
        template,
        stats_recent,
        years=sorted(set(years_all), reverse=True),
        description=description,
    )
    return html


def get_index_list_page(
    template,
    stats,
    years=None,
    description=None,
    prev=None,
    index=None,
    next=None,
):
    # Replace all zeros with "" for the HTML table
    context_stats = []
    for row in stats:
        context_row = {
            key: (val if val != 0 else "") for key, val in zip(row.keys(), row.values())
        }
        context_stats.append(context_row)

    html = template.render(
        kalman_stats=context_stats,
        description=description,
        years=years,
        prev=prev,
        index=index,
        next=next,
    )
    return html


def send_process_mail(opt, evts_perigee):
    subject = "kalman_watch: long drop interval(s)"
    lines = ["Long drop interval(s) found for the following perigee events:"]
    lines.extend(
        f"{evt.dirname} {evt.perigee.date}"
        for evt in evts_perigee
        if len(evt.low_kalmans) > 0
    )
    text = "\n".join(lines)
    send_mail(LOGGER, opt, subject, text, __file__)


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

    def write_info(self, opt):
        """Write info to file"""
        path = EVT_PERIGEE_INFO_PATH(opt.data_dir, self)
        LOGGER.info(f"Writing info to {path}")
        path.write_text(json.dumps(self.info, indent=4))

    def write_data(self, opt):
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

        path = EVT_PERIGEE_DATA_PATH(opt.data_dir, self)
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

    def get_detail_html(self) -> str:
        fig = self.get_plot_fig()
        kalman_plot_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            default_width=1000,
            default_height=600,
        )

        has_low_kalmans = len(self.low_kalmans) > 0
        low_kalmans_html = "\n".join(
            self.low_kalmans.pformat(html=True, max_width=-1, max_lines=-1)
        )
        template = Template(INDEX_DETAIL_PATH().read_text())
        context = {
            "date_perigee": self.perigee.date[:-4],
            "dirname": self.dirname,
            "has_low_kalmans": has_low_kalmans,
            "low_kalmans_html": low_kalmans_html,
            "kalman_plot_html": kalman_plot_html,
            "evt_perigee_prev_dirname": get_dirname(self.prev_date),
            "evt_perigee_next_dirname": get_dirname(self.next_date),
            "obsids": [obs["obsid"] for obs in self.obss],
        }
        html = template.render(**context)  # type: str

        return html

    def make_detail_page(self, opt):
        html = self.get_detail_html()

        dirname_path = EVT_PERIGEE_DIR_PATH(opt.data_dir, self)
        dirname_path.mkdir(parents=True, exist_ok=True)
        path = dirname_path / "index.html"
        LOGGER.info(f"Writing perigee detail page to {path}")
        path.write_text(html)

    def get_plot_fig(self) -> pgo.FigureWidget:
        date_perigee = self.perigee.date[:-4]

        perigee_times = self.data["perigee_times"]
        perigee_times = perigee_times.round(1)
        aokalstr = self.data["aokalstr"]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)  # type: pgo.FigureWidget

        for obs, color in zip(self.obss, cycle(COLOR_CYCLE)):
            obs_tstart_rel = CxoTime(obs["obs_start"]).secs - self.perigee.cxcsec
            obs_tstop_rel = CxoTime(obs["obs_stop"]).secs - self.perigee.cxcsec
            i0, i1 = np.searchsorted(perigee_times, [obs_tstart_rel, obs_tstop_rel])

            obs_times = perigee_times[i0:i1]
            fig.add_trace(
                pgo.Scatter(
                    x=obs_times,
                    y=aokalstr[i0:i1],
                    text=f'Obsid {obs["obsid"]}',
                    name=f'Obsid {obs["obsid"]}',
                    hoverinfo="text",
                    line={"color": color},
                ),
                row=1,
                col=1,
            )

            trace = self.get_ionizing_rad_trace(obs, i0, i1, obs_times)
            fig.add_trace(trace, row=2, col=1)

            trace = self.get_no_tracking_trace(obs, i0, i1, obs_times)
            fig.add_trace(trace, row=2, col=1)

        for low_kalman in self.low_kalmans:
            trace = self.get_low_kalman_trace(low_kalman)
            fig.add_trace(trace, row=1, col=1)

        for axis in range(1, 4):
            trace = self.get_attitude_error_trace(
                perigee_times[::ATT_ERR_SUBSAMP], axis
            )
            fig.add_trace(trace, row=3, col=1)

        # fig.update(layout=layout, row=2, col=1)
        fig.update_yaxes(range=[-0.5, 8.5], row=1, col=1, title_text="AOKALSTR")
        fig.update_yaxes(range=[-0.5, 8.5], row=2, col=1, title_text="Slot")
        fig.update_yaxes(row=3, col=1, title_text="Att Err (arcsec)")
        fig.update_xaxes(title_text=f"Time relative to {date_perigee}", row=2, col=1)

        return fig

    def get_attitude_error_trace(self, times, axis):
        y = np.rad2deg(self.data[f"aoatter{axis}"]) * 3600
        y = y.round(1)
        trace = pgo.Scatter(
            x=times,
            y=y,
            line={"color": px.colors.qualitative.Plotly[axis]},
            name=f"AOATTER{axis}",
        )
        return trace

    def get_low_kalman_trace(self, low_kalman):
        p0 = low_kalman["tstart_rel"]
        p1 = low_kalman["tstop_rel"]
        n_kalstr = low_kalman["n_kalstr"]

        trace = pgo.Scatter(
            x=[p0, p1],
            y=[n_kalstr, n_kalstr],
            marker={"opacity": 0.5},
            line={"color": "red", "width": 3},
            text=f"<= {n_kalstr} stars for {p1-p0:.1f} sec",
            hoverinfo="text",
            showlegend=False,
        )

        return trace

    def get_no_tracking_trace(self, obs, i0, i1, obs_times):
        # Not tracking
        xs = []
        ys = []
        for slot in range(8):
            # fmt: off
            # See: https://github.com/psf/black/issues/236
            bad = (
                ~self.data[f"aca_track{slot}"][i0:i1]
                & self.data["npnt_kalm"][i0:i1]
            )
            # fmt: on
            x = obs_times[bad]
            xs.append(x)
            ys.append(np.full_like(x, fill_value=slot).astype(int))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        trace = pgo.Scatter(
            x=x,
            y=y,
            hoverinfo="text",
            mode="markers",
            line={"color": "#d62728"},
            showlegend=False,
            marker={"opacity": 0.75, "size": 5},
            marker_symbol="x",
            text=f'Obsid {obs["obsid"]}',
        )

        return trace

    def get_ionizing_rad_trace(self, obs, i0, i1, obs_times):
        # Ionizing radiation flag
        xs = []
        ys = []
        for slot in range(8):
            # fmt: off
            bad = (
                self.data[f"aca_ir{slot}"][i0:i1]
                & self.data[f"aca_track{slot}"][i0:i1]
                & self.data["npnt_kalm"][i0:i1]
            )
            # fmt: on

            x = obs_times[bad]
            xs.append(x)
            ys.append(np.full_like(x, fill_value=slot).astype(int))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        trace = pgo.Scatter(
            x=x,
            y=y,
            hoverinfo="text",
            mode="markers",
            line={"color": "#1f77b4"},  # muted blue
            showlegend=False,
            marker={"opacity": 0.75, "size": 5},
            text=f'Obsid {obs["obsid"]}',
        )

        return trace


if __name__ == "__main__":
    import os

    if "PERIGEE_MON_DEVELOP" not in os.environ:
        main()
