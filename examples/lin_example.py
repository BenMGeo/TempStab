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
from scipy import signal

len_years = 5
xs = np.linspace(1,len_years*365,len_years*365)
m = -0.02
t = 3
lin = m*xs+t

yearly = 4.8 * np.sin(np.pi*2*((xs+25)/365.2425))
monthly = 3.1 * np.cos(np.pi*2*((xs+1)/28))
#daily = 1.1 * np.sin(np.pi*2*((xs+50)/7))

noise = np.random.rand(len(xs)) * np.nan
1
while any(np.isnan(noise)):
    for i, _ in enumerate(noise):
        if i == 0:
            noise[i] = np.random.rand(1)
        else:
            A = 3.75
            noise[i] = abs(A*noise[i-1] * (1-noise[i-1]))

the_ts = lin + yearly + monthly + noise*2


#start = np.random.randint(100, 399)
#
#stop = start + np.random.randint(10, 80)
#
#gap = np.arange(start, stop)
#
#the_ts[gap] = np.nan


tdt = datetime.datetime.today()
the_dates = [tdt - datetime.timedelta(days=x) for
             x in range(0, len(the_ts))]
#
#plt.figure()
#plt.plot(the_dates, the_ts)
#plt.show()

TS = TempStab(dates=the_dates, array=the_ts,
              breakpoint_method="this", deseason=True, num_periods=3)

TS.analysis(homogenize=True)

print(TS.periods)

#BP = rdp_bp(TS.numdate, TS.array)
#
#BP = rdp_bp_iter(TS.numdate, TS.array, nIter=25, tol=0.2)

#plt.figure()
#plt.plot(abs(spline_red_array_d2) == min(abs(spline_red_array_d2)))
#plt.show()

plt.figure()
plt.plot(TS.numdate, TS.array, label = "original")
plt.plot(TS.numdate, TS.__trend_removed__, label = "trend")
plt.plot(TS.numdate, TS.array - TS.__trend_removed__, label = "no trend")
plt.plot(TS.numdate, TS.__season_removed__, label = "season")
plt.plot(TS.numdate, TS.__season_removed__ + TS.__trend_removed__, label = "trend + season")
plt.plot(TS.numdate, TS.prep, label = "no trend, no season")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
#plt.plot(BP['rn'], BP['ra'])
#for bp in BP['bp']:
#    plt.axvline(x=bp)
plt.show()

plt.plot(TS.numdate, TS.prep, label = "no trend, no season (?)")
plt.legend()
#
#print(np.diff(BP["bp"]))
#print(np.array(seqs))

P=(signal.convolve(TS.__identified_gaps__,np.array([1,1,1])/3.)[1:-1]!=0)
