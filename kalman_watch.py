import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('agg')
from astropy.table import Table, Column
import jinja2

import matplotlib.pyplot as plt
from Ska.Matplotlib import plot_cxctime
# from kadi import events
from Ska.engarchive import fetch
from Chandra.Time import DateTime
from pyyaks.logger import get_logger


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = get_logger()


def get_opt():
    parser = argparse.ArgumentParser(description='Kalman star watch')
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
    args = parser.parse_args()
    return args


opt = get_opt()

stop = DateTime(opt.stop)
start = DateTime(opt.start or stop - 3 * 365)

logger.info('Getting AOKALSTR between {} and {}'.format(start.date, stop.date))
dat = fetch.Msid('aokalstr', start, stop)
lowkals = dat.logical_intervals('<=', '1 ')
lowkals = lowkals[lowkals['duration'] < 120]

bad = lowkals['duration'] > 60
dt_stop = (stop.secs - DateTime(lowkals['datestart']).secs) / 86400.
recent_bad = bad & (dt_stop < 7)

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

outfile = os.path.join(opt.outdir, 'kalman_drop_intervals.png')
logger.info('Saving plot to {}'.format(os.path.abspath(outfile)))
plt.savefig(outfile)

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
out_html = template.render(long_durs=long_durs[::-1])
with open(os.path.join(opt.outdir, 'index.html'), 'w') as fh:
    fh.write(out_html)
