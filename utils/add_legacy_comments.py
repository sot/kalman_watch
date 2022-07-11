from astropy.table import Table, join

t1 = Table.read('low_kalman_events.ecsv')
comments = """
obsid   comment
------  -------------------------------------------------------------------------------------
24252   SCS107 (Radiation)
25445   SCS107 (HRC Anomaly)
26166   SCS107 (Radiation)
24520   SCS107 (Manual / LETG Anomaly)
22652   Related to HRC Anomaly
21167   HighBgd
22643   HighBgd
16502   Venus
13822   SCS107
14528   SCS107
16065   SCS107
16271   SCS107
16099   SCS107
16344   SCS107
17696   SCS107
19942   BSH
19924   BSH
50095   Reacq at Kalman trans. (5 arcsec thresh)
18975   Reacq at Kalman trans. (5 arcsec thresh)
50062   Reacq at Kalman trans. (5 arcsec thresh)
18997   SCS107
20980   HighBgd
47912   HighBgd
47909   MUPS Checkout
62658   Mixed IRU Setup
"""
t2 = Table.read(comments, format='ascii')

del t1['comment']

t3 = join(t1, t2, keys='obsid', join_type='left')
t3.sort('datestart', reverse=True)
t3.write('low_kalman_events_with_comments.ecsv', format='ascii.ecsv', overwrite=True)