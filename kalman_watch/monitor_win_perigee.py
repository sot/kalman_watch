# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""IR flag fraction from perigee monitor window data (NMAN) and ACA telemetry (NPNT).

Generate a trending plot which combines the perigee monitor window data during NMAN with
the IR flag data (from AOACIIR<n>) during NPNT. This is used to evaluate the
evolution of the high ionizing radiation (IR) zone during perigee.

This is normally run as a script which by default generates a plot
``<data_dir>/mon_win_kalman_drops_-45d_-1d.png`` in the current directory. It will also
cache the MAUDE images in the ``<data_dir>/aca_imgs_cache/`` directory.

NOTE: This script was originally written using AOKALSTR telemetry as a proxy for the IR
flag data. This was later replaced with the more direct AOACIIR<n> telemetry. In many
places there are references to Kalman drops. Generally speaking that should be taken as
a synonym for the IR flag fraction.
"""

import argparse

import re
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

import astropy.units as u

import numpy as np

from cxotime import CxoTime
from astropy.table import vstack
from ska_helpers.logging import basic_logger

from kalman_watch import __version__, OPTIONS
from kalman_watch.kalman_watch_data import (
    NotEnoughImagesError,
    get_manvrs_perigee,
    get_mon_dataset,
    get_kalman_drops_nman,
    get_kalman_drops_npnt,
    get_kalman_drops_prediction,
    clean_aca_images_cache,
    table_from_perigee,
)


logger = basic_logger(__name__, level="INFO")



# Keep track of labels used in the plot so that we don't repeat them
LABELS_USED = {}  # date: label


def get_opt() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor window perigee data {}".format(__version__)
    )
    parser.add_argument(
        "--start",
        type=str,
        default="-45d",
        help='Start date (default=NOW - 45 days written as "-45d")',
    )
    # Default stop is 1 day before now. This is to avoid the case where MAUDE telemetry
    # is not complete through the stop date and the cached images will be incomplete.
    parser.add_argument(
        "--stop", type=str, default="-1d", help="Stop date (default=-1d from now)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=".", help="Data directory (default=.)"
    )
    parser.add_argument(
        "--ir-thresholds-start",
        default="2023:100",
        help="Start date for sampling guide stars for IR thresholds",
    )
    parser.add_argument(
        "--ir-thresholds-stop",
        default="2023:200",
        help="Stop date for sampling guide stars for IR thresholds",
    )
    parser.add_argument(
        "--skip-mon",
        action="store_true",
        help="Skip monitor window data (mostly for testing)",
    )
    parser.add_argument(
        "--n-cache",
        type=int,
        default=70,
        help=(
            "Number of cached ACA images files (~0.7 Mb each) to keep (default=70)"
            " (set to 0 to disable caching)"
        ),
    )
    parser.add_argument("--rad-data", type=str, help="Radiation data from STK")
    return parser


PERIGEE_COLOR_MARKERS_PLOTLY = {}


def get_color_marker_for_perigee_plotly(perigee_date: str) -> tuple[str, str]:
    """Get a color and marker (shape) for a perigee event date.

    Parameters
    ----------
    perigee_date : str
        Perigee event date

    Returns
    -------
    color : str
        Color for perigee event
    marker : str
        Marker for perigee event
    """
    if perigee_date not in PERIGEE_COLOR_MARKERS_PLOTLY:
        n_perigees = len(PERIGEE_COLOR_MARKERS_PLOTLY)
        colors = px.colors.qualitative.D3
        markers = [
            'circle',
            'square',
            'diamond',
            'cross',
            'x',
            'triangle-up',
            'triangle-down',
            'triangle-left',
            'triangle-right',
            ]
        color = colors[n_perigees % len(colors)]
        marker = markers[n_perigees % len(markers)]
        PERIGEE_COLOR_MARKERS_PLOTLY[perigee_date] = (color, marker)
        logger.info(f"Perigee {perigee_date} {color=} {marker=}")
    return PERIGEE_COLOR_MARKERS_PLOTLY[perigee_date]


def short_date(date: str):
    """Shorten a date string to just the day of year and time.

    2024:056:12:34:56.789 -> 056 1234z
    012345678901234567890
    """
    return f"{date[5:8]} {date[9:11]}{date[12:14]}z"


def plot_mon_win_and_aokalstr_composite_plotly(
    kalman_drops_npnt_list, kalman_drops_nman_list, kalman_drops_prediction_list, outfile=None, title=""
):
    """Plot the monitor window (NMAN) and NPNT IR flags fraction data.

    Parameters
    ----------
    kalman_drops_npnt : KalmanDropsData
        Output of get_kalman_drops_npnt
    kalman_drops_nman_list : list[KalmanDropsData]
        Output of get_kalman_drops_nman
    kalman_drops_prediction_list : list[KalmanDropsData]
        Output of get_kalman_drops_prediction
    outfile : str
        Output file name (default=None)
    title : str
        Title for plot (default="")

    Returns
    -------
    None
    """
    if outfile is None:
        outfile = "kalman_plot.html"

    fig = go.FigureWidget()

    if not title:
        title = f"IR flag fraction near perigee {kalman_drops_npnt_list.start.iso[:7]}"

    tables = []
    if len(kalman_drops_nman_list) > 0:
        # this happens at times before the monitoring windows were implemented
        nman_table = vstack([table_from_perigee(kalman_drops) for kalman_drops in kalman_drops_nman_list if len(kalman_drops.times) > 0])
        nman_table["type"] = 0
        tables.append(nman_table)
    npnt_table = vstack([table_from_perigee(kalman_drops) for kalman_drops in kalman_drops_npnt_list if len(kalman_drops.times) > 0])
    npnt_table["type"] = 1
    tables.append(npnt_table)
    table = vstack(tables)
    # type 0 first, because that determines the marker in the legend
    table.sort(["type", "cxcsec"])

    traces = {}

    perigee_dates = np.unique(table["perigee"])

    for perigee_date in perigee_dates:
        color, marker = get_color_marker_for_perigee_plotly(perigee_date)
        ok = table["perigee"] == perigee_date
        markers = np.where(table["type"][ok] == 0, marker, f"{marker}-open")
        marker_size = np.where(table["type"][ok] == 0, 5, 5)
        line_width = np.where(table["type"][ok] == 0, 0, 1)
        traces[perigee_date] = {
            "times": table["time"][ok] / 60,
            "values": table["kalman_drops"][ok],
            "colors": [color] * len(table["time"][ok]),
            "marker": markers,
            "marker_size": marker_size,
            "line_width": line_width,
        }

    plotly_traces = []
    show_model_on_start = False

    logger.info("plotting predictions")
    for idx, pred_win_kalman_drops in enumerate(kalman_drops_prediction_list):
        perigee_date = pred_win_kalman_drops.perigee_date
        color, marker = get_color_marker_for_perigee_plotly(pred_win_kalman_drops.perigee_date.date)
        if len(pred_win_kalman_drops.times) > 0:
            plotly_traces.append(f"pred-{perigee_date}")
            fig.add_trace(
                go.Scattergl(
                    x=pred_win_kalman_drops.times / 60,
                    y=pred_win_kalman_drops.kalman_drops.clip(None, 160),
                    line={"color": color},
                    name=f"pred {perigee_date}",
                    visible=show_model_on_start,
                    legendgroup=f"{perigee_date}",
                    showlegend=False,
                )
            )

    for perigee_date, trace in traces.items():
        plotly_traces.append(f"obs-{perigee_date}")
        fig.add_trace(
            go.Scattergl(
                x=trace["times"],
                y=trace["values"],
                marker={
                    "size": trace["marker_size"],
                    "color": trace["colors"],
                    "opacity": 0.7,
                    "symbol": trace["marker"],
                    "line_width": trace["line_width"],
                },
                mode="markers",
                name=f"{perigee_date[5:8]} {perigee_date[9:11]}{perigee_date[12:14]}z",
                legendgroup=perigee_date,
                legendgrouptitle_text="",
            )
        )

    fig.update_layout(
        title=title,
        template="seaborn",
        xaxis_title="Time from perigee (minutes)",
        autosize=False,
        width=1000,
        height=600,
        xaxis = {
            "tickmode": 'linear',
            "tick0": -100,
            "dtick": 10,
        },
        xaxis_range=[-110, 110],
        yaxis_range=[-0.05, 1.05],
        margin={"b": 200},
    )

    fig.update_layout(
        legend= {
            "x": 0.,
            "y": -0.2,
            "yanchor": "top",
            "orientation": "h",
            "font": {"size": 9},
        },
    )

    predicted_traces = [idx for idx, trace in enumerate(plotly_traces) if trace.startswith("pred-")]

    fig.update_layout(
        updatemenus=[
            dict(
                type = "buttons",
                direction = "left",
                active=0 if show_model_on_start else 1,
                buttons=list([
                    dict(
                        args2=[
                            {"visible": [False] * len(predicted_traces)},
                            {},
                            predicted_traces,
                        ],
                        args=[
                            {"visible": [True] * len(predicted_traces)},
                            {},
                            predicted_traces,
                        ],
                        label="Show Rad Model",
                        method="update"
                    )
                ]),
                pad={"r": 0, "t": 0},
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.96,
                yanchor="top"
            ),
        ]
    )

    if outfile:
        kalman_plot_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            default_width=1000,
            default_height=600,
            config={"displayModeBar": True},
        )
        with open(outfile, "w") as fh:
            fh.write(kalman_plot_html)


def cxotime_reldate(date):
    r"""Parse a date string that can contain a relative time and return a CxoTime.

    The relative time matches this regex: ``[-+]? [0-9]* \.? [0-9]+ d``. Examples
    include ``-45d`` and ``+1.5d`` (-45 days and +1.5 days from now, respectively).

    Parameters
    ----------
    date : str
        Date string

    Returns
    -------
    out : CxoTime
        CxoTime object
    """
    if re.match(r"[-+]? [0-9]* \.? [0-9]+ d", date, re.VERBOSE):
        dt = float(date[:-1])
        out = CxoTime.now() + dt * u.d
    else:
        out = CxoTime(date)
    return out


def main(args=None):
    opt = get_opt().parse_args(args)
    start = cxotime_reldate(opt.start)
    stop = cxotime_reldate(opt.stop)

    OPTIONS.data_dir = opt.data_dir
    OPTIONS.ir_thresholds_start = opt.ir_thresholds_start
    OPTIONS.ir_thresholds_stop = opt.ir_thresholds_stop
    if opt.rad_data is not None:
        OPTIONS.rad_table_path = opt.rad_data

    # Intervals of NMAN within 100 minutes of perigee
    manvrs_perigee = [] if opt.skip_mon else get_manvrs_perigee(start, stop)

    # Get list of monitor window data for each perigee maneuver
    mons = []
    for manvr in manvrs_perigee:
        try:
            mon = get_mon_dataset(
                manvr["datestart"],
                manvr["datestop"],
                opt.ir_thresholds_start,
                opt.ir_thresholds_stop,
                opt.data_dir,
                cache=opt.n_cache > 0,
            )
            mons.append(mon)
        except NotEnoughImagesError as err:
            logger.warning(err)

    # Process monitor window (NMAN) data into kalman drops per minute for each maneuver.
    # This uses idx to assign a different color to each maneuver (in practice each
    # perigee).
    logger.info("processing NMAN data")
    kalman_drops_nman_list = [get_kalman_drops_nman(mon) for mon in mons]

    # Process NPNT data for the entire time range into kalman drops per minute. This
    # assigns different colors to each perigee.
    kalman_drops_npnt_list = get_kalman_drops_npnt(start, stop)

    kalman_drops_prediction_list = get_kalman_drops_prediction(start, stop)

    outfile = Path(opt.data_dir) / "kalman_plot.html"
    title = f"IR flag fraction {start.date[:8]} to {stop.date[:8]}"

    plot_mon_win_and_aokalstr_composite_plotly(
        kalman_drops_npnt_list,
        kalman_drops_nman_list,
        kalman_drops_prediction_list,
        outfile=outfile,
        title=title
    )

    clean_aca_images_cache(opt.n_cache, opt.data_dir)


if __name__ == "__main__":
    main()
