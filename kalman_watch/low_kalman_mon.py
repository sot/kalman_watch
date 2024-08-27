# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Process telemetry and find intervals of low Kalman stars.

- Intervals are stored in <data_dir>/low_kalman_events.ecsv.
- Web dashboard is stored in <data_dir>/index_lowkals.html.
"""

import argparse
from pathlib import Path

import astropy.units as u
import jinja2
import numpy as np
import plotly.graph_objects as pgo
from astropy.table import Table, vstack
from cxotime import CxoTime
from Ska.engarchive import fetch
from Ska.engarchive.utils import logical_intervals
from ska_helpers.logging import basic_logger

from . import __version__

# Constants and file path definitions
FILE_DIR = Path(__file__).parent


def LOWKALS_DATA_PATH(data_dir: str) -> Path:
    return Path(data_dir) / "low_kalman_events.ecsv"


def LOWKALS_HTML_PATH(data_dir: str) -> Path:
    return Path(data_dir) / "index.html"


def INDEX_TEMPLATE_PATH():
    return FILE_DIR / "index_low_kalman_template.html"


LOWKALS_EMPTY = Table(
    names=["datestart", "duration", "obsid", "comment"],
    dtype=["S21", "float", "int", "S40"],
)


logger = basic_logger(__name__, level="INFO")


def get_opt():
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
    parser.add_argument(
        "--long-duration",
        type=float,
        default=27.0,
        help="Threshold for long duration drop intervals (secs, default=27.0)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="Threshold for writing/plotting drop intervals (secs, default=4.0 sec)",
    )
    parser.add_argument(
        "--highlight-recent-days",
        type=float,
        default=30.0,
        help="Number of days to highlight in plots and table (days, default=30)",
    )
    parser.add_argument(
        "--in-file",
        default="mon_win_kalman_drops_-45d_-1d.html",
        help="Input file with monitor-window data. Default: mon_win_kalman_drops_-45d_-1d.html",
    )
    return parser


def main(sys_args=None):
    opt = get_opt().parse_args(sys_args)

    lowkals_path = LOWKALS_DATA_PATH(opt.data_dir)
    lowkals_path.parent.mkdir(exist_ok=True, parents=True)
    lowkals_prev = get_lowkals_prev(lowkals_path)

    # Start lookback days from stop, except don't start before the last telemetry
    stop = CxoTime(opt.stop)
    start = stop - opt.lookback * u.day  # type: CxoTime
    date_telem_last = CxoTime(lowkals_prev.meta.pop("date_telem_last", "1999:001"))
    if start < date_telem_last:
        logger.info("Overriding starting time with the time of latest low-kalman event")
        start = date_telem_last

    if stop <= start:
        logger.error(f"Starting time happens after stopping time ({start.date} > {stop.date})")
        return 1

    lowkals_new = get_lowkals_new(opt, start, stop, date_telem_last)
    lowkals = vstack([lowkals_new, lowkals_prev])  # type: Table
    lowkals.sort("datestart", reverse=True)
    logger.info(f"Updating events file {lowkals_path}")
    lowkals.write(lowkals_path, format="ascii.ecsv", overwrite=True)

    make_web_page(opt, lowkals)


def get_lowkals_prev(lowkals_path: Path) -> Table:
    """Get the existing low kalman events or make a new one"""
    if lowkals_path.exists():
        lowkals_prev = Table.read(lowkals_path)
    else:
        lowkals_prev = LOWKALS_EMPTY.copy()
    return lowkals_prev


def get_lowkals_new(
    opt, start: CxoTime, stop: CxoTime, date_telem_last: CxoTime
) -> Table:
    """Get low Kalman events from telemetry"""
    # Get the AOKALSTR data with number of kalman stars reported by OBC.
    logger.info(f"Getting telemetry between {start} and {stop}")

    print(start, stop)
    dat = fetch.Msidset(["aokalstr", "aoacaseq", "aopcadmd", "cobsrqid"], start, stop)
    dat.interpolate(1.025)

    if len(dat.times) < 300:
        logger.warning("WARNING: Not enough data to find low Kalman intervals")
        lowkals = LOWKALS_EMPTY.copy()
        lowkals.meta["date_telem_last"] = date_telem_last.date
        return lowkals

    logger.info("Finding intervals of low kalman stars")
    # Find intervals of low kalman stars. We use complete_intervals=True to ensure that
    # only complete intervals get stored. A partial interval will get picked up in the
    # next daily processing.
    lowkals = logical_intervals(
        dat["aokalstr"].times,
        (dat["aokalstr"].vals.astype(int) <= 1)
        & (dat["aoacaseq"].vals == "KALM")
        & (dat["aopcadmd"].vals == "NPNT"),
        max_gap=10.0,
        complete_intervals=True,
    )

    # Select events with minimum duration (4 seconds by default)
    lowkals = lowkals[lowkals["duration"] > opt.min_duration]

    # Find the obsid and add an empty comment column
    ii = np.searchsorted(dat["cobsrqid"].times, lowkals["tstart"])
    lowkals["obsid"] = dat["cobsrqid"].vals[ii].astype(int)
    lowkals["comment"] = np.full(len(lowkals), "")
    lowkals["duration"] = np.round(lowkals["duration"], 1)

    del lowkals["datestop"]
    del lowkals["tstart"]
    del lowkals["tstop"]

    # Warn in processing for long duration drop intervals
    long_mask = lowkals["duration"] > opt.long_duration
    for lowkal in lowkals[long_mask]:
        logger.warning(
            f"WARNING: Fewer than two kalman stars at {lowkal['datestart']} "
            f"for {lowkal['duration']:.1f} secs"
        )

    # Provide log information and store last telemetry date in table meta.
    date_telem_last = CxoTime(dat["aokalstr"].times[-1]).date
    logger.info(f"Last telemetry date: {date_telem_last}")
    lowkals.meta["date_telem_last"] = date_telem_last
    if (n_lowkals := len(lowkals)) > 0:
        logger.info(f"Found {n_lowkals} new low kalman events")

    return lowkals


def get_plot_html(opt, lowkals: Table, show=False) -> str:
    dates = CxoTime(lowkals["datestart"])
    times = dates.datetime64
    text_obsids = np.array(
        [f"{lowkal['datestart'][:-4]} ObsID {lowkal['obsid']}" for lowkal in lowkals]
    )

    fig = pgo.Figure()
    recent = dates > CxoTime.now() - opt.highlight_recent_days * u.day
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
            "title": (
                "Duration of contiguous n_kalman <= 1 (autoscale for full mission)"
            ),
            "yaxis": {"title": "Duration (sec)", "autorange": False, "range": [0, 35]},
            "xaxis": {
                "title": "Date",
                "range": [
                    (CxoTime.now() - 5 * 365 * u.day).datetime,
                    CxoTime.now().datetime,
                ],
                "autorange": False,
            },
        }
    )
    if show:
        fig.show()

    html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        default_width=800,
        default_height=500,
    )
    return html


def make_web_page(opt, lowkals: Table) -> None:
    # Setup for the web report template rendering
    ok = (lowkals["duration"] > opt.long_duration) & (
        CxoTime.now() - CxoTime(lowkals["datestart"]) < 5 * u.year
    )
    long_durs = lowkals[ok]

    tr_classes = []
    for lowkal in long_durs:
        recent = (
            CxoTime.now() - CxoTime(lowkal["datestart"])
            < opt.highlight_recent_days * u.day
        )
        tr_class = 'class="pink-bkg"' if recent else ""
        tr_classes.append(tr_class)
    long_durs["tr_class"] = tr_classes

    with open(Path(opt.data_dir) / opt.in_file) as f:
        kalman_plot_html = f.read()

    index_template_html = INDEX_TEMPLATE_PATH().read_text()
    template = jinja2.Template(index_template_html)
    out_html = template.render(
        long_durs=long_durs,
        long_dur_limit=opt.long_duration,
        last_date=lowkals.meta["date_telem_last"][:-4],
        plot_html=get_plot_html(opt, lowkals),
        kalman_plot_html=kalman_plot_html,
    )
    lowkals_html_path = LOWKALS_HTML_PATH(opt.data_dir)
    logger.info(f"Writing HTML to {lowkals_html_path}")
    lowkals_html_path.write_text(out_html)


if __name__ == "__main__":
    main()
