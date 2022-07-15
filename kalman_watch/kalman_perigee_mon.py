import argparse
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Union, List, Tuple

import astropy.units as u
import numpy as np
import plotly.graph_objects as pgo
from astropy.table import Table
from cheta.fetch import MSIDset
from cheta.utils import logical_intervals
from cxotime import CxoTime
from jinja2 import Template
from kadi.commands import get_observations
from kadi.commands.commands_v2 import get_cmds
from plotly.subplots import make_subplots
from ska_helpers.logging import basic_logger

from kalman_watch import __version__

logger = basic_logger(__name__, level="INFO")


class NotEnoughTelemetry(ValueError):
    pass


FILE_DIR = Path(__file__).parent


def PERIGEES_DIR_PATH(data_dir: str) -> Path:
    return Path(data_dir) / "perigees"


def EVT_PERIGEE_DIR_PATH(data_dir: str, evt: "EventPerigee"):
    return PERIGEES_DIR_PATH(data_dir) / evt.dirname


def DASHBOARD_TEMPLATE_PATH():
    return FILE_DIR / "index_dashboard_template.html"


class CxoTimeField:
    def __init__(self, *, default=None):
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        setattr(obj, self._name, CxoTime(value))


@dataclass
class EventPerigee:
    """Class for tracking Kalman star data through perigee."""

    rad_entry: CxoTimeField = CxoTimeField()
    perigee: CxoTimeField = CxoTimeField()
    rad_exit: CxoTimeField = CxoTimeField()
    prev_date: str = ""
    next_date: str = ""

    def __repr__(self):
        return (
            f"EventPerigee(rad_entry={self.rad_entry.date!r},"
            f" perigee={self.perigee.date!r},"
            f" rad_exit={self.rad_exit.date!r})"
        )

    @property
    def dirname(self):
        return get_dirname(self.perigee.date)

    @property
    def obss(self):
        if not hasattr(self, "_obss"):
            self._obss = get_observations(start=self.rad_entry, stop=self.rad_exit)
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
            "aonstars",
            "aopcadmd",
            "aoacaseq",
            "aoaciir*",
            "aoacrpt",
            "aoacfct*",
        ]
        # TODO: use MAUDE backorbit telemetry
        tlm = MSIDset(msids, self.rad_entry, self.rad_exit)
        if (
            len(tlm["aokalstr"]) == 0
            or self.rad_exit.cxcsec - tlm["aokalstr"].times[-1] > 100
        ):
            return None

        for msid in ["aokalstr", "aonstars"]:
            tlm[msid].vals = tlm[msid].vals.astype(np.float64)

        # Reduce everything to the first ACA values during NPNT/KALM
        ok = tlm["aoacrpt"].vals == "0 "
        tlm.interpolate(
            times=tlm["aokalstr"].times[ok], bad_union=False, filter_bad=False
        )

        for msid in ["aokalstr", "aonstars"]:
            bad = (
                (tlm["aopcadmd"].vals != "NPNT")
                | (tlm["aoacaseq"].vals != "KALM")
                | tlm[msid].bads
            )
            tlm[msid].vals[bad] = np.nan

        tlm.perigee_times = tlm.times - self.perigee.cxcsec

        return tlm

    @property
    def low_kalmans(self):
        if not hasattr(self, "_low_kalmans"):
            self._low_kalmans = self._get_low_kalmans()
        return self._low_kalmans

    def _get_low_kalmans(self):
        rows = []
        for n_kalstr, dur_limit in KALMAN_LIMITS:
            vals = self.tlm["aokalstr"].vals.copy()
            vals[np.isnan(vals)] = 10
            ints_low = logical_intervals(self.tlm["aokalstr"].times, vals <= n_kalstr)
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

        low_kalmans = Table(rows=rows)
        for col in low_kalmans.itercols():
            if col.info.dtype.kind == "f":
                col.info.format = ".1f"

        return low_kalmans

    def get_dashboard_page_html(self):
        kalman_plot_html = self.get_plot_html()
        has_low_kalmans = len(self.low_kalmans) > 0
        low_kalmans_html = "\n".join(
            self.low_kalmans.pformat(
                html=True, max_width=-1, max_lines=-1, tableclass="sample"
            )
        )
        template = Template(Path("index_kalman_template.html").read_text())
        context = {
            "date_perigee": self.perigee.date[:-4],
            "has_low_kalmans": has_low_kalmans,
            "low_kalmans_html": low_kalmans_html,
            "kalman_plot_html": kalman_plot_html,
            "evt_perigee_prev": get_dirname(self.prev_date),
            "evt_perigee_next": get_dirname(self.next_date),
            "obsids": [obs["obsid"] for obs in self.obss],
        }
        html = template.render(**context)

        return html

    def get_plot_html(self, show=False):
        date_perigee = self.perigee.date[:-4]

        perigee_times = self.tlm.perigee_times
        perigee_times = perigee_times.round(2)
        aokalstr = self.tlm["aokalstr"].vals.astype(float)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
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

            # Ionizing radiation flag
            xs = []
            ys = []
            for slot in range(8):
                bad = (
                    (self.tlm[f"aoaciir{slot}"].vals[i0:i1] == "ERR")
                    & (self.tlm[f"aoacfct{slot}"].vals[i0:i1] == "TRAK")
                    & (self.tlm["aopcadmd"].vals[i0:i1] == "NPNT")
                    & (self.tlm["aoacaseq"].vals[i0:i1] == "KALM")
                )

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
            fig.add_trace(trace, row=2, col=1)

            # Not tracking
            xs = []
            ys = []
            for slot in range(8):
                bad = (
                    (self.tlm[f"aoacfct{slot}"].vals[i0:i1] != "TRAK")
                    & (self.tlm["aopcadmd"].vals[i0:i1] == "NPNT")
                    & (self.tlm["aoacaseq"].vals[i0:i1] == "KALM")
                )

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
            fig.add_trace(trace, row=2, col=1)

        for low_kalman in self.low_kalmans:
            p0 = low_kalman["tstart_rel"]
            p1 = low_kalman["tstop_rel"]
            n_kalstr = low_kalman["n_kalstr"]

            fig.add_trace(
                pgo.Scatter(
                    x=[p0, p1],
                    y=[n_kalstr, n_kalstr],
                    marker={"opacity": 0.5},
                    line={"color": "red", "width": 3},
                    text=f"<= {n_kalstr} stars for {p1-p0:.1f} sec",
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # fig.update(layout=layout, row=2, col=1)
        fig.update_yaxes(range=[-0.5, 8.5], row=1, col=1, title_text="AOKALSTR")
        fig.update_yaxes(range=[-0.5, 8.5], row=2, col=1, title_text="Slot")
        fig.update_xaxes(title_text=f"Time relative to {date_perigee}", row=2, col=1)

        if show:
            fig.show()

        html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            default_width=1000,
            default_height=600,
        )

        return html


def get_dirname(date: str) -> str:
    out = date[:17].replace(":", "_") if date else ""
    return out


def get_opt(sys_args):
    parser = argparse.ArgumentParser(
        description="Kalman star watch {}".format(__version__)
    )
    parser.add_argument("--stop", type=str, help="Stop date (default=NOW)")
    parser.add_argument(
        "--lookback",
        type=float,
        default=14,
        help="Lookback days from stop for processing (days, default=14)",
    )
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    args = parser.parse_args(sys_args)
    return args


def main(sys_args):
    opt = get_opt(sys_args)

    stop = CxoTime(opt.stop)
    start = stop - opt.lookback * u.day

    evts_perigee = get_evts_perigee(start, stop)

    dirnames, n_low_kalmans = make_html_pages(opt, evts_perigee)

    make_index_list_page(dirnames, n_low_kalmans)


def make_html_pages(
    opt: argparse.Namespace, evts_perigee: List[EventPerigee]
) -> Tuple[List[str], int]:
    dirnames = []
    n_low_kalmans = []

    for evt_perigee in evts_perigee:
        html, low_kalmans = evt_perigee.make_html_page()

        dirname_path = EVT_PERIGEE_DIR_PATH(opt.data_dir)
        dirname_path.mkdir(parents=True, exist_ok=True)
        (dirname_path / "index.html").write_text(html)

        dirnames.append(get_dirname(evt_perigee))
        n_low_kalmans.append(len(low_kalmans))

    return dirnames, n_low_kalmans


def get_evts_perigee(start, stop):
    """
    Get the perigee EEF1000, EPERIGEE and XEF1000 fully within start/stop

    :param start: CxoTime-like
        Start of date range
    :param stop: CxoTime-like
        End of date range
    :returns: list of PerigeeEvent
        List of PerigeeEvent objects
    """
    event_types = ["EEF1000", "EPERIGEE", "XEF1000"]
    cmds = get_cmds(start=start, stop=stop, type="ORBPOINT")
    ok = np.isin(cmds["event_type"], event_types)
    cmds = cmds[ok]

    # Find index in cmds of EEF1000 commands
    idxs_rad_entry = np.where(cmds["event_type"] == event_types[0])[0]
    if len(idxs_rad_entry) == 0:
        return []

    # Iterate through commands starting from first rad entry in sets of 3 for
    # rad entry, perigee and rad exit.
    events = []
    cmds = cmds[idxs_rad_entry[0] :]
    for cmd0, cmd1, cmd2 in zip(cmds[0::3], cmds[1::3], cmds[2::3]):
        cmds_event_types = [cmd["event_type"] for cmd in [cmd0, cmd1, cmd2]]
        if cmds_event_types != event_types:
            raise ValueError(f"Expected {event_types} but got {cmds_event_types}")

        event = EventPerigee(
            rad_entry=cmd0["date"], perigee=cmd1["date"], rad_exit=cmd2["date"]
        )
        if event.tlm is not None:
            events.append(event)
        else:
            break

    for evt_prev, evt, evt_next in zip(
        [None] + events[:-1], events, events[1:] + [None]
    ):
        evt.prev_date = "" if evt_prev is None else evt_prev.perigee.date
        evt.next_date = "" if evt_next is None else evt_next.perigee.date

    return events


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


def make_index_list_page(opt, dirnames, n_low_kalmans):
    template = Template(Path("index_list_template.html").read_text())
    context = {
        "values": list(zip(reversed(dirnames), reversed(n_low_kalmans))),
    }
    html = template.render(**context)
    (PERIGEES_DIR_PATH(opt.data_dir) / "index.html").write_text(html)


def make_plot_plotly(lowkals):
    dates = CxoTime(lowkals["datestart"])
    times = dates.datetime64
    text_obsids = np.array(
        [f"{lowkal['datestart'][:-4]} ObsID {lowkal['obsid']}" for lowkal in lowkals]
    )

    layout = {
        "title": f"Duration of contiguous n_kalman <= 1",
        "yaxis": {"title": "Duration (sec)"},
        "xaxis": {"title": f"Date"},
        "yaxis_range": [0, 35],
    }

    fig = pgo.Figure(layout=layout)
    recent = dates > CxoTime.now() - 30 * u.day
    for color, mask in [
        ("#1f77b4", ~recent),  # muted blue
        ("#ff7f0e", recent),  # safety orange
    ]:
        trace = pgo.Scatter(
            x=times[mask],
            y=lowkals["duration"][mask].clip(None, 32),
            hoverinfo="text",
            mode="markers",
            line={"color": color},
            showlegend=False,
            marker={"opacity": 0.75, "size": 8},
            text=text_obsids[mask],
        )
        fig.add_trace(trace)

    fig.update_layout(
        {
            "xaxis_autorange": False,
            "xaxis_range": [
                (CxoTime.now() - 5 * 365 * u.day).datetime,
                CxoTime.now().datetime,
            ],
        }
    )
    fig.show()
    fig.write_html(
        "test_plotly.html",
        full_html=False,
        include_plotlyjs="cdn",
        default_width=1000,
        default_height=600,
    )