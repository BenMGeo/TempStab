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

xs = np.linspace(1,500,5000)
m = 0.02
t = 3
lin = m*xs+t

yearly = 4.8 * np.sin(np.pi*2*((xs+25)/331))
monthly = 3.1 * np.cos(np.pi*2*((xs+1)/71))
daily =1.1 * np.sin(np.pi*2*((xs+8)/6))

noise = (np.random.rand(len(xs)) - 0.5) * 3

the_ts = lin + yearly + monthly + daily + noise


#start = np.random.randint(100, 399)
#
#stop = start + np.random.randint(10, 80)
#
#gap = np.arange(start, stop)
#
#the_ts[gap] = np.nan


tdt = datetime.datetime.today()
the_dates = [tdt - datetime.timedelta(days=x, minutes=7.3345*x) for
             x in range(0, len(the_ts))]

plt.figure()
plt.plot(the_dates, the_ts)
plt.show()

TS = TempStab(dates=the_dates, array=the_ts,
              breakpoint_method="this", detrend=True)

TS.analysis(homogenize=True)

#BP = rdp_bp(TS.numdate, TS.array)
#
#BP = rdp_bp_iter(TS.numdate, TS.array, nIter=25, tol=0.2)

#plt.figure()
#plt.plot(abs(spline_red_array_d2) == min(abs(spline_red_array_d2)))
#plt.show()

plt.figure()
plt.plot(TS.numdate, TS.array)
plt.plot(TS.numdate, TS.prep)
#plt.plot(BP['rn'], BP['ra'])
#for bp in BP['bp']:
#    plt.axvline(x=bp)
plt.show()
#
#print(np.diff(BP["bp"]))
#print(np.array(seqs))

P=(signal.convolve(TS.__identified_gaps__,np.array([1,1,1])/3.)[1:-1]!=0)
