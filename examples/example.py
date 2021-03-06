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

seqs = np.random.randint(50,150,4)
seqs = seqs.tolist()
seqs.append(500-sum(seqs))
if min(seqs)<0:
    seqs =  [i - min(seqs) for i in seqs]

bias = np.random.randint(-10,10,5) * 1.
bias[3] = np.NAN
#bias[1] = np.NAN

array = [bias[i] + np.random.rand(seqs[i]) for i in range(len(seqs))]

the_ts = np.concatenate(array)

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
