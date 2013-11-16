#!/usr/bin/env python

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('agg')
from astropy.table import Table, Column
import jinja2

import matplotlib.pyplot as plt
from Ska.Matplotlib import plot_cxctime
from Ska.engarchive import fetch
from Chandra.Time import DateTime
from pyyaks.logger import get_logger


VERSION = '0.2'
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = get_logger()


def get_opt():
    parser = argparse.ArgumentParser(description='Kalman star watch {}'.format(VERSION))
    parser.add_argument('--start',
                        type=str,
                        help='Start date')
    parser.add_argument('--stop',
                        type=str,
                        help='Stop date')
    parser.add_argument('--outdir',
                        type=str,
                        default='.',
                        help='Output directory')
    parser.add_argument('--long-duration',
                        type=float,
                        default=60.0,
                        help='Threshold for long duration drop intervals (default=60 sec)')
    args = parser.parse_args()
    return args


opt = get_opt()

stop = DateTime(opt.stop)
start = DateTime(opt.start or stop - 3 * 365)

# Get the AOKALSTR data with number of kalman stars reported by OBC
logger.info('Getting AOKALSTR between {} and {}'.format(start.date, stop.date))
dat = fetch.Msid('aokalstr', start, stop)
last_date = DateTime(dat.times[-1]).date

logger.info('Finding intervals of low kalman stars')
# Find intervals of low kalman stars
lowkals = dat.logical_intervals('<=', '1 ')

# Very long intervals are spurious (need to understand this fully)
lowkals = lowkals[lowkals['duration'] < 120]

# Select long-duration events
bad = lowkals['duration'] > opt.long_duration
dt_stop = (stop.secs - DateTime(lowkals['datestart']).secs) / 86400.
recent_bad = bad & (dt_stop < 7)

# Store any new long duration events to a persistent shelf database
events_file = os.path.join(opt.outdir, 'long_dur_events.shelve')
logger.info('Storing new events to {}'.format(events_file))
import shelve
bad_db = shelve.open(events_file)
for long_dur in lowkals[bad]:
    datestart = long_dur['datestart']
    if datestart not in bad_db:
        logger.warn('WARNING: Fewer than two kalman stars at {} for {:.1f} secs'
                    .format(datestart, long_dur['duration']))
        bad_db[datestart] = long_dur['duration']
bad_db.close()

# Make the plot
plt.figure(figsize=(6, 4))
plot_cxctime(lowkals['tstart'], lowkals['duration'], '.b')
if np.any(recent_bad):
    plot_cxctime(lowkals['tstart'][recent_bad], lowkals['duration'][recent_bad], '.r', markersize=8)
x0, x1 = plt.xlim()
dx = (x1 - x0) * 0.05
plt.xlim(x0 - dx, x1 + dx)
plt.ylim(-5, 100)
plt.grid()
plt.ylabel('Duration (seconds)')
plt.title('Duration of contiguous n_kalman <= 1')

# Plot a line for the time of the OBC patch to ignore the defective pixel flag.
t_dp = DateTime('2013-10-24T12:00:00').secs
if t_dp > start.secs:
    plot_cxctime([t_dp, t_dp], [-5, 100], '--r', alpha=0.4)

outfile = os.path.join(opt.outdir, 'kalman_drop_intervals.png')
logger.info('Saving plot to {}'.format(os.path.abspath(outfile)))
plt.savefig(outfile)

# Setup for the web report template rendering
long_durs = lowkals[bad]
long_durs = Table(long_durs)
long_durs.add_column(Column(name='recent', data=dt_stop[bad] < 7))

obsids = []
tr_classes = []
for long_dur in long_durs:
    obsdat = fetch.Msid('cobsrqid', long_dur['datestart'], long_dur['datestop'])
    obsids.append(obsdat.vals[0])
    tr_class = ('class="pink-bkg"' if long_dur['recent'] else '')
    tr_classes.append(tr_class)
long_durs.add_column(Column(name='tr_class', data=tr_classes))
long_durs.add_column(Column(name='obsid', data=obsids, dtype=int))
long_durs['duration'] = np.round(long_durs['duration'], 1)

index_template_html = open(os.path.join(FILE_DIR, 'index_template.html'), 'r').read()
template = jinja2.Template(index_template_html)
out_html = template.render(long_durs=long_durs[::-1], long_dur_limit=opt.long_duration,
                           last_date=last_date)
with open(os.path.join(opt.outdir, 'index.html'), 'w') as fh:
    fh.write(out_html)
