#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("agg")
import astropy.units as u
import jinja2
import matplotlib.pyplot as plt
import plotly.graph_objects as pgo
from astropy.io import ascii
from astropy.table import Column, Table, vstack
from cxotime import CxoTime
from pyyaks.logger import get_logger
from Ska.engarchive import fetch
from Ska.engarchive.utils import logical_intervals
from Ska.Matplotlib import plot_cxctime

from . import __version__

FILE_DIR = Path(__file__).parent


def LOWKALS_PATH(data_dir: str) -> Path:
    return Path(data_dir) / "low_kalman_events.ecsv"


logger = get_logger()


def get_opt(sys_args):
    parser = argparse.ArgumentParser(
        description="Kalman star watch {}".format(__version__)
    )
    parser.add_argument("--stop", type=str, help="Stop date (default=NOW)")
    parser.add_argument(
        "--lookback",
        type=float,
        default=14,
        help="Lookback time from stop (days, default=14)",
    )
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    parser.add_argument(
        "--long-duration",
        type=float,
        default=27.0,
        help="Threshold for long duration drop intervals (default=27.0 sec)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="Threshold for plotting drop intervals (default=4.0 sec)",
    )
    args = parser.parse_args(sys_args)
    return args


def main(sys_args=None):
    opt = get_opt(sys_args)

    lowkals_path = LOWKALS_PATH(opt.data_dir)
    lowkals_prev = get_lowkals_prev(lowkals_path)

    # Start lookback days from stop, except don't start before the last telemetry
    stop = CxoTime(opt.stop)
    start = stop - opt.lookback * u.day
    date_telem_last = CxoTime(lowkals_prev.meta.pop("date_telem_last", "1999:001"))
    if start < date_telem_last:
        start = date_telem_last

    lowkals_new = get_lowkals_new(opt.min_duration, opt.long_duration, start, stop)
    if len(lowkals_new) > 0:
        logger.info(f"Storing new events to {lowkals_path}")
        lowkals = vstack([lowkals_new, lowkals_prev])
        lowkals.sort("datestart", reverse=True)
        lowkals.write(lowkals_path, format="ascii.ecsv", overwrite=True)
    else:
        lowkals = lowkals_prev

    make_web_page(opt, lowkals)


def get_lowkals_prev(lowkals_path: Path) -> Table:
    # Get the existing low kalman events or make a new one
    if lowkals_path.exists():
        lowkals_prev = Table.read(lowkals_path)
    else:
        lowkals_prev = Table(
            names=["datestart", "duration", "obsid", "comment"],
            dtype=["S21", "float", "int", "S40"],
        )
    return lowkals_prev


def get_lowkals_new(
    min_duration: float, long_duration: float, start: CxoTime, stop: CxoTime
) -> Table:
    # Get the AOKALSTR data with number of kalman stars reported by OBC
    logger.info(f"Getting AOKALSTR between {start} and {stop}")
    dat = fetch.Msidset(["aokalstr", "aoacaseq", "aopcadmd", "cobsrqid"], start, stop)
    dat.interpolate(1.025)

    logger.info("Finding intervals of low kalman stars")
    # Find intervals of low kalman stars
    lowkals = logical_intervals(
        dat["aokalstr"].times,
        (dat["aokalstr"].vals.astype(int) <= 1)
        & (dat["aoacaseq"].vals == "KALM")
        & (dat["aopcadmd"].vals == "NPNT"),
        max_gap=10.0,
    )

    # Select events with minimum duration (not necessarily "long")
    lowkals = lowkals[lowkals["duration"] > min_duration]

    ii = np.searchsorted(dat["cobsrqid"].times, lowkals["tstart"])
    lowkals["obsid"] = dat["cobsrqid"].vals[ii].astype(int)
    lowkals["comment"] = np.full(len(lowkals), "")
    lowkals["duration"] = np.round(lowkals["duration"], 1)

    del lowkals["datestop"]
    del lowkals["tstart"]
    del lowkals["tstop"]

    # Warn in processing for long duration drop intervals
    long_mask = lowkals["duration"] > long_duration
    for lowkal in lowkals[long_mask]:
        logger.warn(
            f"WARNING: Fewer than two kalman stars at {lowkal['datestart']} "
            f"for {lowkal['duration']:.1f} secs"
        )

    lowkals.meta["date_telem_last"] = CxoTime(dat["aokalstr"].times[-1]).date
    return lowkals


def get_plot_html(lowkals: Table) -> str:
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
    html = fig.to_html(
        full_html=False, include_plotlyjs="cdn", default_width=1000, default_height=600,
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
        recent = CxoTime.now() - CxoTime(lowkal["datestart"]) < 30 * u.day
        tr_class = 'class="pink-bkg"' if recent else ""
        tr_classes.append(tr_class)
    long_durs["tr_class"] = tr_classes

    index_template_html = (FILE_DIR / "index_reacqs_template.html").read_text()
    template = jinja2.Template(index_template_html)
    out_html = template.render(
        long_durs=long_durs[::-1],
        long_dur_limit=opt.long_duration,
        last_date=lowkals.meta["date_telem_last"][:-4],
        plot_html=get_plot_html(lowkals),
    )
    (Path(opt.data_dir) / "index.html").write_text(out_html)


if __name__ == "__main__":
    main()
