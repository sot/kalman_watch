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
import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

os.environ["MPLBACKEND"] = "Agg"

import astropy.units as u
import kadi.events
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import scipy.signal
from astropy.table import Table, vstack
from chandra_aca.maude_decom import get_aca_images
from chandra_aca.planets import get_earth_blocks
from chandra_aca.transform import mag_to_count_rate
from cheta import fetch
from cheta.utils import logical_intervals
from cxotime import CxoTime, CxoTimeLike
from kadi.commands import get_cmds, get_observations, get_starcats
from ska_helpers.logging import basic_logger

from kalman_watch import __version__
from kalman_watch.kalman_perigee_mon import EventPerigee

matplotlib.style.use("bmh")

logger = basic_logger(__name__, level="INFO")


# Typing hint for a table of images coming from chandra_aca.maude_decom.get_aca_images()
ACAImagesTable: TypeAlias = Table
MonDataSet: TypeAlias = dict[str]  # TODO: use dataclass


@dataclass
class KalmanDropsData:
    start: CxoTime
    stop: CxoTime
    times: np.ndarray
    kalman_drops: np.ndarray
    colors: list


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
        default=30,
        help=(
            "Number of cached ACA images files (~0.7 Mb each) to keep (default=30)"
            " (set to 0 to disable caching)"
        ),
    )
    return parser


def get_manvrs_perigee(start: CxoTimeLike, stop: CxoTimeLike) -> Table:
    """Get maneuvers that start or stop within 40 minutes of perigee.

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
        Table of intervals of maneuvers that are entirely within 40 minutes of perigee.
    """
    start = CxoTime(start)
    stop = CxoTime(stop)

    logger.info(f"Getting maneuvers from {start.date} to {stop.date}")
    cmds = get_cmds(start, stop)
    perigee_dates = cmds[cmds["event_type"] == "EPERIGEE"]["date"]
    pcad_mode = fetch.Msid("aopcadmd", start, stop)

    # TODO: with the "intervals" refactor WIP this could be done more nicely by taking
    # the intersection of the maneuvers and the Earth block intervals. Instead here we
    # stub in a 3rd state "EART" for Earth block intervals and report every interval
    # even though almost all of them are not during a perigee maneuver.
    logger.info(f"Getting intervals of Earth blocks from {start.date} to {stop.date}")
    blocks = get_earth_blocks(start, stop)
    for block in blocks:
        logger.info(
            f" Excluding Earth block from {block['datestart']} "
            f"to {block['datestop']}"
        )
        idx0, idx1 = pcad_mode.times.searchsorted([block["tstart"], block["tstop"]])
        pcad_mode.vals[idx0:idx1] = "EART"

    manvrs = logical_intervals(pcad_mode.times, pcad_mode.vals == "NMAN")

    # Get maneuvers that are entirely within 40 minutes of perigee
    manvrs_perigee = []
    for perigee_date in CxoTime(perigee_dates):
        ok = (np.abs(manvrs["tstart"] - perigee_date.secs) < 40 * 60) | (
            np.abs(manvrs["tstop"] - perigee_date.secs) < 40 * 60
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
    logger.info(f"Getting ACA images from {start.date} to {stop.date}")
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
    logger.info(f"Cleaning ACA images cache to keep most recent {n_cache} files")
    data_dir = Path(data_dir)
    cache_dir = data_dir / "aca_imgs_cache"

    # Keep the most recent n_cache files based on file creation time
    cache_files = sorted(
        cache_dir.glob("aca_imgs_*.fits.gz"),
        key=lambda x: x.stat().st_mtime,
    )
    for path in cache_files[:-n_cache]:
        logger.info(f"Deleting {path}")
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
    logger.info(f"Getting IR thresholds from {start} to {stop}")
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

    hits = Table(hits)
    hits["hit_idx"] = np.arange(len(hits))

    return hits


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
    logger.info(f"Getting MON data from {start.date} to {stop.date}")
    imgs = get_aca_images_cached(start, stop, data_dir, cache=cache)

    # Create a dataset of MON data for this slot
    mon = {}
    mons = []
    for slot in range(8):
        try:
            mons.append(process_imgs(imgs, slot))
        except NotEnoughImagesError as err:
            logger.warning(err)
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


@functools.lru_cache()
def get_stk_pred() -> Table:
    """Get the predicted IR fraction from STK.

    Returns
    -------
    stk_pred : Table
        Table of predicted IR fraction from STK
    """
    stk_pred = Table.read("stk_pred_ir_frac.ecsv")
    return stk_pred


def get_kalman_drops_nman(mon: MonDataSet, idx: int):
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
    color = ["r", "m", "b", "g", "k"][idx % 5]
    colors = [color] * len(dt_mins)
    times = np.array(dt_mins) * 60
    kalman_drops_data = KalmanDropsData(
        mon["start"], mon["stop"], times, kalman_drops, colors
    )
    stk_pred = get_stk_pred()
    idxs = np.searchsorted(stk_pred["time"], times + mon["perigee_date"].secs)
    kalman_drops_data.stk_ir_frac = stk_pred["ir_frac"][idxs]
    return kalman_drops_data


def _reshape_to_n_sample_2d(arr: np.ndarray, n_sample: int = 60) -> np.ndarray:
    """Reshape 1D array to 2D with one row per n_sample samples."""
    arr = arr[: -(arr.size % n_sample)]
    arr = arr.reshape(-1, n_sample)
    return arr


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


def get_kalman_drops_npnt(start, stop, duration=100) -> KalmanDropsData:
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
    kalman_drops_data : KalmanDropsData
    """
    start = CxoTime(start)
    stop = CxoTime(stop)
    rad_zones = kadi.events.rad_zones.filter(start, stop).table
    perigee_times = CxoTime(rad_zones["perigee"])
    event_perigees = get_perigee_events(perigee_times, duration)
    times_from_perigee_list = []
    n_drops_list = []
    colors_list = []
    for idx, ep in enumerate(event_perigees):
        if len(ep.data["times"]) > 200:
            times_from_perigee, n_drops = get_binned_drops_from_event_perigee(ep)
            times_from_perigee_list.append(times_from_perigee)
            n_drops_list.append(n_drops)
            colors_list.append([f"C{idx}"] * len(times_from_perigee))

    return KalmanDropsData(
        start,
        stop,
        np.concatenate(times_from_perigee_list),
        np.concatenate(n_drops_list),
        np.concatenate(colors_list),
    )


def plot_kalman_drops(
    kalman_drops_data: KalmanDropsData,
    ax,
    alpha: float = 1.0,
    title: str | None = None,
    marker_size: float = 10,
) -> matplotlib.collections.PathCollection:
    """Plot the fraction of IR flags set per minute.

    Parameters
    ----------
    kalman_drops_data : KalmanDropsData
        Output of get_kalman_drops_nman or get_kalman_drops_npnt
    ax : matplotlib.axes.Axes
        Matplotlib axes
    alpha : float
        Alpha value for scatter plot
    title : str, None
        Title for plot (default=None)
    marker_size : float
        Marker size for scatter plot (default=10)

    Returns
    -------
    scat : matplotlib.collections.PathCollection
        Scatter plot collection
    """
    scat = ax.scatter(
        kalman_drops_data.times / 60,
        kalman_drops_data.kalman_drops.clip(None, 160),
        s=marker_size,
        c=kalman_drops_data.colors,
        alpha=alpha,
    )
    if hasattr(kalman_drops_data, "stk_ir_frac"):
        ax.plot(
            kalman_drops_data.times / 60,
            kalman_drops_data.stk_ir_frac,
            color=kalman_drops_data.colors[0],
        )
    # set major ticks every 10 minutes
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    if title is None:
        title = f"IR flag fraction near perigee {kalman_drops_data.start.iso[:7]}"
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time from perigee (minutes)")
    return scat


def plot_mon_win_and_aokalstr_composite(
    kalman_drops_npnt, kalman_drops_nman_list, outfile=None, title=""
):
    """Plot the monitor window (NMAN) and NPNT IR flags fraction data.

    Parameters
    ----------
    kalman_drops_npnt : KalmanDropsData
        Output of get_kalman_drops_npnt
    kalman_drops_nman_list : list[KalmanDropsData]
        Output of get_kalman_drops_nman
    outfile : str
        Output file name (default=None)
    title : str
        Title for plot (default="")

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))

    plot_kalman_drops(
        kalman_drops_npnt,
        ax=ax,
        alpha=0.4,
    )

    for mon_win_kalman_drops in kalman_drops_nman_list:
        plot_kalman_drops(
            mon_win_kalman_drops, ax=ax, alpha=1.0, marker_size=20, title=""
        )

    ax.set_title(title)
    ax.set_ylim(-0.05, 1.0)
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=200)


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

    # Intervals of NMAN within 40 minutes of perigee
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
    kalman_drops_nman_list: list[KalmanDropsData] = []
    mon_times_list = []
    mon_ir_fracs_list = []
    for idx, mon in enumerate(mons):
        kalman_drops_nman = get_kalman_drops_nman(mon, idx)
        kalman_drops_nman_list.append(kalman_drops_nman)
        mon_times_list.append(kalman_drops_nman.times + mon["perigee_date"].secs)
        mon_ir_fracs_list.append(kalman_drops_nman.kalman_drops)

    # Process NPNT data for the entire time range into kalman drops per minute. This
    # assigns different colors to each perigee.
    kalman_drops_npnt = get_kalman_drops_npnt(start, stop)

    outfile = Path(opt.data_dir) / f"mon_win_kalman_drops_{opt.start}_{opt.stop}.png"
    title = f"IR flag fraction {start.date[:8]} to {stop.date[:8]}"
    plot_mon_win_and_aokalstr_composite(
        kalman_drops_npnt, kalman_drops_nman_list, outfile=outfile, title=title
    )

    mon_times = np.concatenate(mon_times_list)
    mon_ir_fracs = np.concatenate(mon_ir_fracs_list)
    outfile = Path(opt.data_dir) / f"mon_win_kalman_drops_{opt.start}_{opt.stop}.ecsv"
    dat = Table([mon_times, mon_ir_fracs], names=["time", "ir_frac"])
    dat.write(outfile, format="ascii.ecsv", overwrite=True)

    clean_aca_images_cache(opt.n_cache, opt.data_dir)


if __name__ == "__main__":
    main()
