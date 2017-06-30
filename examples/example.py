#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:53:49 2017

@author: bmueller
"""
import numpy as np
import sys
import datetime
import matplotlib.pyplot as plt
sys.path.append("/media/bmueller/Work/GIT/TempStab/")
from TempStab import TempStab
from TempStab import rdp
from TempStab import rdp_bp, rdp_bp_iter
from scipy import interpolate

the_ts = np.concatenate(
        (3 + np.random.rand(100),
         12 + np.random.rand(100),
         6 + np.random.rand(100),
         8 + np.random.rand(100),
         4 + np.random.rand(100))
        )

# the_ts = 3 + np.random.rand(100)
the_dates = [datetime.datetime.today() -
             datetime.timedelta(days=x, minutes=7.3345*x) for
             x in range(0, len(the_ts))]

plt.figure()
plt.plot(the_dates, the_ts)
plt.show()

TS = TempStab(dates=the_dates, array=the_ts,
              breakpoint_method="this", detrend=True)

BP = rdp_bp(TS.numdate, TS.array)

rdp_bp_iter(TS.numdate, TS.array)

#plt.figure()
#plt.plot(abs(spline_red_array_d2) == min(abs(spline_red_array_d2)))
#plt.show()

plt.figure()
plt.plot(TS.numdate, TS.array)
plt.plot(TS.numdate, TS.prep)
plt.plot(BP['rn'], BP['ra'])
for bp in BP['bp']:
    plt.axvline(x=bp)
plt.show()
