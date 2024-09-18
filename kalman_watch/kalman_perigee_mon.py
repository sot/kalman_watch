# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Watch Kalman star data during perigee passages."""

import argparse
import os
from itertools import cycle
from pathlib import Path

import astropy.units as u
import numpy as np
import plotly.express as px
import plotly.graph_objects as pgo
from acdc.common import send_mail
from astropy.table import Table, vstack
from cxotime import CxoTime
from jinja2 import Template
from plotly.subplots import make_subplots
from ska_helpers.logging import basic_logger
from ska_helpers.run_info import log_run_info

from kalman_watch import __version__, OPTIONS, paths
from kalman_watch.kalman_watch_data import (
    get_dirname,
    EventPerigee,
    get_stats,
    get_evts_perigee,
    read_kalman_stats,
)

LOGGER = basic_logger(__name__, level="INFO")


FILE_DIR = Path(__file__).parent

# Sub-sample attitude error by 8 to reduce the number of points.
ATT_ERR_SUBSAMP = 8


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


def get_opt() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kalman star watch {}".format(__version__)
    )
    parser.add_argument("--stop", type=str, help="Stop date (default=NOW)")
    parser.add_argument(
        "--lookback", type=float, default=14, help="Lookback time (days, default=14)"
    )
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    parser.add_argument("--rad-data", type=str, help="Radiation data from STK")
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

    OPTIONS.data_dir = opt.data_dir
    if opt.rad_data is not None:
        OPTIONS.rad_table_path = opt.rad_data

    stop = CxoTime(opt.stop)
    start: CxoTime = stop - opt.lookback * u.day

    stats_prev = read_kalman_stats()
    evts_perigee = get_evts_perigee(start, stop, stats_prev)

    # Bail out if there are no new perigee events. Since we backed up to the
    # previous perigee we need at least two perigee events.
    if len(evts_perigee) == 0:
        LOGGER.info("No new perigee events, exiting")
        return

    for evt_perigee in evts_perigee:
        evt_perigee.write_data()
        evt_perigee.write_info()

    stats_new = get_stats(evts_perigee)
    # Combine new and previous statistics summary
    stats_all = vstack([stats_new, stats_prev])
    stats_all.sort("perigee", reverse=True)

    path = paths.perigees_index_table_path()
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
            evt_tmp = EventPerigeeMon(stat["rad_entry"], stat["perigee"], stat["rad_exit"])
            evt = EventPerigeeMon.from_npz(evt_tmp.data_path)
            evt.prev_date = date_prev
            evt.next_date = date_next
            evt.make_detail_page()

        make_index_list_pages(stats_all)


def make_index_list_pages(stats_all: Table) -> None:
    html = get_index_html_recent(stats_all)
    path = paths.perigees_dir_path() / "index.html"
    LOGGER.info(f"Writing recent index list page to {path}")
    path.write_text(html)

    # Write index page for the last two calendar years
    years_all = CxoTime(stats_all["perigee"]).ymdhms.year
    years_unique = sorted(np.unique(years_all))
    for year in years_unique[-2:]:
        html = get_index_html_year(stats_all, year)
        path = paths.perigees_dir_path() / str(year) / "index.html"
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


class EventPerigeeMon(EventPerigee):
    """Class for tracking Kalman star data through perigee."""

    def __init__(self, rad_entry, perigee, rad_exit):
        super().__init__(rad_entry, perigee, rad_exit)
        pass

    def get_detail_html(self) -> str:
        fig = self.get_plot_fig()
        kalman_plot_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            default_width=1000,
            default_height=800,
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

    def make_detail_page(self):
        html = self.get_detail_html()

        dirname_path = self.dir_path
        dirname_path.mkdir(parents=True, exist_ok=True)
        path = dirname_path / "index.html"
        LOGGER.info(f"Writing perigee detail page to {path}")
        path.write_text(html)

    def get_plot_fig(self) -> pgo.FigureWidget:
        date_perigee = self.perigee.date[:-4]

        perigee_times = self.data["perigee_times"]
        perigee_times = perigee_times.round(1)
        aokalstr = self.data["aokalstr"]

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1)  # type: pgo.FigureWidget

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

        fig.add_trace(
            pgo.Scatter(
                x=self.predicted_kalman_drops["times"],
                y=self.predicted_kalman_drops["values"],
                name="Rad. model",
                line={"color": "black"},
            ),
            row=4,
            col=1
        )
        kalman_drops_nman = self.get_kalman_drops_nman()
        fig.add_trace(
            pgo.Scatter(
                x=kalman_drops_nman["times"],
                y=kalman_drops_nman["values"],
                name="NMAN",
                mode="markers",
                marker={"color": px.colors.qualitative.Plotly[0], "size": 5},
            ),
            row=4,
            col=1
        )
        kalman_drops_npnt = self.get_kalman_drops_npnt()
        fig.add_trace(
            pgo.Scatter(
                x=kalman_drops_npnt["times"],
                y=kalman_drops_npnt["values"],
                name="NPNT",
                mode="markers",
                marker={"color": px.colors.qualitative.Plotly[1], "size": 5},
            ),
            row=4,
            col=1
        )

        # fig.update(layout=layout, row=2, col=1)
        fig.update_yaxes(range=[-0.5, 8.5], row=1, col=1, title_text="AOKALSTR")
        fig.update_yaxes(range=[-0.5, 8.5], row=2, col=1, title_text="Slot")
        fig.update_yaxes(row=3, col=1, title_text="Att Err (arcsec)")
        fig.update_yaxes(range=[-0.1, 1.1], row=4, col=1, title_text="IR flag rate")
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
