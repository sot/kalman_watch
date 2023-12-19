# # Perigee monitor window data and kalman drops
#
# This notebook examines the monitor window data for the first handful of perigees after
# the start of monitor windows during maneuvers.
#
# It also massages the data into an effective number of kalman drops per minute, matching
# roughly the same quantity which has been shown in the kalman drops evolution analysis.
# The data are plotted with the same function as in kalman-drops-perigee-evolution.ipynb.

import argparse
import functools
from pathlib import Path
import re

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import scipy.signal
from astropy.table import Table, vstack
from chandra_aca.maude_decom import get_aca_images
from chandra_aca.transform import mag_to_count_rate
from cheta import fetch
from cheta.utils import logical_intervals
from cxotime import CxoTime, CxoTimeLike, date2secs
from kadi.commands import get_cmds, get_observations, get_starcats
from ska_helpers.logging import basic_logger
import kadi.events

from kalman_watch import __version__

matplotlib.style.use("bmh")

logger = basic_logger(__name__, level="INFO")


# TODO: replace with generic Earth block code in chandra_aca.planets (coded in
# 2023-11 quarterly ATM).
EARTH_BLOCKS = [
    ("2023:297:12:23:00", "2023:297:12:48:37"),
    ("2023:300:03:49:00", "2023:300:04:16:40"),
]


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
        "--stop", type=str, default="-1d", help='Stop date (default=-1d from now)'
    )
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory (default=.)")
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

    for block_start, block_stop in EARTH_BLOCKS:
        idx0, idx1 = pcad_mode.times.searchsorted(
            [date2secs(block_start), date2secs(block_stop)]
        )
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


def get_aca_images_cached(start: CxoTimeLike, stop: CxoTimeLike, data_dir: str | Path):
    """Get ACA images from MAUDE and cache them in a file.

    Images are cached in ``data_dir/aca_img_cache/``. Files outside of start/stop range
    are deleted.

    Parameters
    ----------
    start : CxoTimeLike
        Start time
    stop : CxoTimeLike
        Stop time

    Returns
    -------
    imgs : Table
        Table of ACA image data from chandra_aca.maude_decom.get_aca_images()

    """
    logger.info(f"Getting ACA images from {start.date} to {stop.date}")
    datestart = CxoTime(start).date
    datestop = CxoTime(stop).date
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)
    filename = f"aca_imgs_{datestart}_{datestop}.fits.gz"
    cache = cache_dir / filename
    if cache.exists():
        out = Table.read(cache)
    else:
        imgs = get_aca_images(datestart, datestop)
        imgs.write(cache)
        out = imgs

    # Delete files outside of start/stop range
    for filename in cache_dir.glob("aca_imgs_*.fits.gz"):
        datestart, datestop = filename.name.split("_")[2:4]
        if CxoTime(datestart) < start or CxoTime(datestop) > stop:
            filename.unlink()

    return out


def process_imgs(imgs: Table, slot: int) -> Table:
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
def get_ir_thresholds(start, stop):
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


def get_hits(mon):
    ir_thresholds_iter = get_ir_thresholds("2023:100", "2023:200")

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


def get_mon_dataset(start, stop):
    start = CxoTime(start)
    stop = CxoTime(stop)
    start.format = "date"
    stop.format = "date"
    logger.info(f"Getting MON data from {start.date} to {stop.date}")
    imgs = get_aca_images_cached(start, stop)

    # Create a dataset of MON data for this slot
    mon = {}
    mons = []
    for slot in range(8):
        mons.append(process_imgs(imgs, slot))
    mon["imgs"] = vstack(mons)
    mon["imgs"].sort("TIME")
    mon["imgs"]["img_idx"] = np.arange(len(mon["imgs"]))
    mon["perigee_date"] = get_nearest_perigee_date(imgs["TIME"][len(imgs) // 2])
    mon["imgs"]["dt_min"] = (mon["imgs"]["TIME"] - mon["perigee_date"].secs) / 60
    mon["start"] = start
    mon["stop"] = stop

    mon["hits"] = get_hits(mon)

    return mon


def get_kalman_drops_per_minute(mon):
    # Get the number of drops per "minute", where a "minute" is really 60 * 1.025
    # seconds, or 1.025 minutes. This matches what is done in
    # kalman-drops-perigee-evolution.ipynb.
    bins = np.arange(-30, 31, 1.025)
    kalman_drops = []
    dt_mins = []
    for dt_min0, dt_min1 in zip(bins[:-1], bins[1:]):
        ok = (mon["hits"]["dt_min"] >= dt_min0) & (mon["hits"]["dt_min"] < dt_min1)
        n_drops = np.sum(mon["hits"][ok]["ir_flag"])
        if n_drops > 0:
            kalman_drops.append(n_drops)
            dt_mins.append((dt_min0 + dt_min1) / 2)

    return np.array(dt_mins), np.array(kalman_drops)


# Copied from kalman-drops-perigee-evolution.ipynb but added `label` argument
def plot_kalman_drops_old(kalman_drops_data, ax, alpha=1.0, title=None, label=None):
    _, _, times, n_drops, colors = kalman_drops_data
    scat = ax.scatter(
        times / 60,
        n_drops.clip(None, 160),
        s=10,
        c=colors,
        alpha=alpha,
        label=label,
    )
    # set major ticks every 10 minutes
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_xlabel("Time from perigee (minutes)")
    return scat


def get_kalman_drops_data(mon, idx):
    """Get kalman_drops data in the peculiar form for plot_kalman_drops"""
    dt_mins, kalman_drops = get_kalman_drops_per_minute(mon)
    color = ["r", "m", "b", "g", "k"][idx % 5]
    colors = [color] * len(dt_mins)
    times = np.array(dt_mins) * 60
    kalman_drops_data = (mon["start"], mon["stop"], times, kalman_drops, colors)
    return kalman_drops_data


def plot_kalman_drops_all(kalman_drops_data_list, savefig=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
    for kalman_drops_data in kalman_drops_data_list:
        plot_kalman_drops_old(
            kalman_drops_data,
            ax,
            alpha=1.0,
        )
    ax.set_title("Kalman drops per minute")

    if savefig:
        fig.savefig(savefig)


def reshape_to_n_sample_2d(arr, n_sample=60):
    arr = arr[: -(arr.size % n_sample)]
    arr = arr.reshape(-1, n_sample)
    return arr


def process_dat(dat, n_sample=60):
    dat = dat.interpolate(times=dat["aokalstr"].times, copy=True)
    n_kalmans = dat["aokalstr"].raw_vals.astype(float)
    times = dat.times
    ok = (dat["aopcadmd"].vals == "NPNT") & (dat["aoacaseq"].vals == "KALM")
    n_kalmans[~ok] = np.nan
    # Resize n_kalmans to be a multiple of n_sample and then reshape to be 2D with one
    # row per 60 samples
    n_kalmans = reshape_to_n_sample_2d(n_kalmans, n_sample)
    times = reshape_to_n_sample_2d(times, n_sample)
    n_lost_means = np.nansum(8 - n_kalmans, axis=1)
    time_means = np.nanmean(times, axis=1)
    # Count the number of nans in each row
    n_nans = np.sum(np.isnan(n_kalmans), axis=1)
    ok = n_nans == 0

    return time_means[ok], n_lost_means[ok]


def get_kalman_drops(start, stop, duration=100):
    start = CxoTime(start)
    stop = CxoTime(stop)
    rad_zones = kadi.events.rad_zones.filter(start, stop).table
    perigee_times = CxoTime(rad_zones["perigee"])
    msids = ["aokalstr", "aopcadmd", "aoacaseq"]
    times_list = []
    n_drops_list = []
    colors_list = []
    for idx, perigee_time in enumerate(perigee_times):
        dat = fetch.MSIDset(
            msids, perigee_time - duration * u.min, perigee_time + duration * u.min
        )
        if len(dat["aokalstr"]) > 200:
            times, n_drops = process_dat(dat)
            times_list.append(times - perigee_time.secs)
            n_drops_list.append(n_drops)
            colors_list.append([f"C{idx}"] * len(times))

    return (
        start,
        stop,
        np.concatenate(times_list),
        np.concatenate(n_drops_list),
        np.concatenate(colors_list),
    )


def plot_kalman_drops(kalman_drops_data, ax, alpha=1.0, title=None, marker_size=10):
    start, stop, times, n_drops, colors = kalman_drops_data
    scat = ax.scatter(
        times / 60,
        n_drops.clip(None, 160),
        s=marker_size,
        c=colors,
        alpha=alpha,
    )
    # set major ticks every 10 minutes
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    if title is None:
        title = f"Kalman drops per minute near perigee {start.iso[:7]}"
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time from perigee (minutes)")
    return scat


def plot_mon_win_and_aokalstr_composite(
    kalman_drops_npnt, kalman_drops_nman_list, outfile=None, title=""
):
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
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile)


def cxotime_reldate(date):
    if re.match("[-+]? [0-9]* \.? [0-9]+ d", date, re.VERBOSE):
        dt = float(date[:-1])
        out = CxoTime.now() + dt * u.d
    else:
        out = CxoTime(date)
    return out


def main(args=None):
    opt = get_opt().parse_args(args)
    start = cxotime_reldate(opt.start)
    stop = cxotime_reldate(opt.stop)

    manvrs_perigee = get_manvrs_perigee(start, stop)

    mons = []
    for manvr in manvrs_perigee:
        mon = get_mon_dataset(manvr["datestart"], manvr["datestop"])
        mons.append(mon)

    kalman_drops_nman_list = []
    for idx, mon in enumerate(mons):
        kalman_drops_data = get_kalman_drops_data(mon, idx)
        kalman_drops_nman_list.append(kalman_drops_data)

    kalman_drops_npnt = get_kalman_drops(start, stop)

    outfile = Path(opt.data_dir) / f"mon_win_kalman_drops_{opt.start}_{opt.stop}.png"
    title = f"Perigee Kalman drops per minute {start.date[:8]} to {stop.date[:8]}"
    plot_mon_win_and_aokalstr_composite(
        kalman_drops_npnt, kalman_drops_nman_list, outfile=outfile, title=title
    )


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()