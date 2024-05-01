import kalman_watch.monitor_win_perigee as mwp
import argparse
from ska_helpers.logging import basic_logger
from astropy.table import Table, vstack

logger = basic_logger(__name__, level="INFO")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor window perigee data {}".format(mwp.__version__)
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
    return parser


def get_data(opt):
    start = opt.start
    stop = opt.stop

    # Intervals of NMAN within 40 minutes of perigee
    manvrs_perigee = mwp.get_manvrs_perigee(start, stop)

    # Get list of monitor window data for each perigee maneuver
    mons = []
    for manvr in manvrs_perigee:
        try:
            mon = mwp.get_mon_dataset(
                manvr["datestart"],
                manvr["datestop"],
                opt.ir_thresholds_start,
                opt.ir_thresholds_stop,
                opt.data_dir,
                cache=True,
            )
            mons.append(mon)
        except mwp.NotEnoughImagesError as err:
            logger.warning(err)

    # Process monitor window (NMAN) data into kalman drops per minute for each maneuver.
    # This uses idx to assign a different color to each maneuver (in practice each
    # perigee).
    kalman_drops_nman_list: list[mwp.KalmanDropsData] = []
    for idx, mon in enumerate(mons):
        kalman_drops_nman = mwp.get_kalman_drops_nman(mon, idx)
        kalman_drops_nman_list.append(kalman_drops_nman)

    return manvrs_perigee, mon, kalman_drops_nman_list


def main():
    opt = get_parser().parse_args()
    manvrs_perigee, _, kalman_drops_nman_list = get_data(opt)
    kalman_drops_tables = []
    for kd in kalman_drops_nman_list:
        t = Table()
        t['time'] = kd.times
        t['kalman_drops'] = kd.kalman_drops
        t['datestart'] = kd.start.date
        t['datestop'] = kd.stop.date
        kalman_drops_tables.append(t)
    
    kalman_drops_table = vstack(kalman_drops_tables)
    kalman_drops_table.write("kalman_drops.fits", overwrite=True)
    manvrs_perigee.write("manvrs_perigee.fits", overwrite=True)
    logger.info("Wrote kalman_drops.fits and manvrs_perigee.fits")


if __name__ == "__main__":
    main()
