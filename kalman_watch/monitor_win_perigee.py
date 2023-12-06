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
import pickle
from pathlib import Path

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
from cxotime import CxoTime, date2secs
from kadi.commands import get_cmds, get_observations, get_starcats
from ska_helpers.logging import basic_logger

from kalman_watch import __version__

matplotlib.use("Agg")
matplotlib.style.use("bmh")

LOGGER = basic_logger(__name__, level="INFO")


EARTH_BLOCKS = [
    ("2023:297:12:23:00", "2023:297:12:48:37"),
    ("2023:300:03:49:00", "2023:300:04:16:40"),
]


def get_opt() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor window perigee data {}".format(__version__)
    )
    parser.add_argument("--stop", type=str, help="Stop date (default=NOW)")
    parser.add_argument(
        "--lookback", type=float, default=14, help="Lookback time (days, default=14)"
    )
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    return parser


def get_manvrs_perigee(start, stop):
    """Get maneuvers that are entirely within 40 minutes of perigee.

    This is used to select the monitor window data to compute the number of kalman drops
    per minute during perigee.

    Returns
    -------
    manvrs_perigee : Table
        Table of intervals of maneuvers that are entirely within 40 minutes of perigee.
    """
    start = CxoTime(start)
    stop = CxoTime(stop)
    LOGGER.info(f"Getting maneuvers from {start.date} to {stop.date}")

    cmds = get_cmds(start, stop)

    perigee_dates = cmds[cmds["event_type"] == "EPERIGEE"]["date"]

    with fetch.data_source("cxc", "maude"):
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
        ok = (np.abs(manvrs["tstart"] - perigee_date.secs) < 40 * 60) & (
            np.abs(manvrs["tstop"] - perigee_date.secs) < 40 * 60
        )
        manvrs_perigee.append(manvrs[ok])

    manvrs_perigee = vstack(manvrs_perigee)

    return manvrs_perigee


def get_aca_images_cached(start, stop):
    LOGGER.info(f"Getting ACA images from {start.date} to {stop.date}")
    datestart = CxoTime(start).date
    datestop = CxoTime(stop).date
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"aca_imgs_{datestart}_{datestop}.fits.gz"
    if cache.exists():
        return Table.read(cache)
    else:
        imgs = get_aca_images(datestart, datestop)
        imgs.write(cache)
        return imgs


def process_imgs(imgs, slot):
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


def get_nearest_perigee_date(date) -> CxoTime:
    date = CxoTime(date)
    cmds = get_cmds(date - 3 * u.d, date + 3 * u.d, event_type="EPERIGEE")
    idx = np.argmin(np.abs(cmds["time"] - date.secs))
    return CxoTime(cmds["date"][idx])


@functools.lru_cache()
def get_ir_thresholds(start, stop):
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
    LOGGER.info(f"Getting MON data from {start.date} to {stop.date}")
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
def plot_kalman_drops(kalman_drops_data, ax, alpha=1.0, title=None, label=None):
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
        plot_kalman_drops(
            kalman_drops_data,
            ax,
            alpha=1.0,
        )
    ax.set_title("Kalman drops per minute")

    if savefig:
        fig.savefig(savefig)


def main():
    opt = get_opt().parse_args()
    stop = CxoTime(opt.stop)
    start = stop - opt.lookback * u.d

    manvrs_perigee = get_manvrs_perigee(start, stop)

    mons = []
    for manvr in manvrs_perigee:
        mon = get_mon_dataset(manvr["datestart"], manvr["datestop"])
        mons.append(mon)

    kalman_drops_data_list = []
    for idx, mon in enumerate(mons):
        kalman_drops_data = get_kalman_drops_data(mon, idx)
        kalman_drops_data_list.append(kalman_drops_data)

    outfile = Path(opt.data_dir) / "mon_win_kalman_drops.png"
    plot_kalman_drops_all(kalman_drops_data_list, savefig=outfile)

    # pickle.dump(kalman_drops_data_list, open("mon_win_kalman_drops_data.pkl", "wb"))


if __name__ == "__main__":
    main()
