#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:53:49 2017

@author: bmueller
"""

# in some editors, the editor might not load the full environment 
import os
os.environ["R_HOME"] = '/home/bmueller/anaconda2/lib/R'

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

seas1 = 4.8 * np.sin(np.pi*2*((xs+25)/365.2425))
seas2 = 3.1 * np.cos(np.pi*2*((xs+1)/28))
seas3 = 1.1 * np.sin(np.pi*2*((xs+50)/4560))

noise = np.random.rand(len(xs)) * np.nan

while any(np.isnan(noise)):
    for i, _ in enumerate(noise):
        if i == 0:
            noise[i] = np.random.rand(1)
        else:
            A = np.random.rand(1)*(4.-np.pi)+np.pi
            noise[i] = abs(A*noise[i-1] * (1-noise[i-1]))

the_ts = lin + seas1 + seas2 + seas3 + noise*5-12

#start = np.random.randint(100, 399)
#
#stop = start + np.random.randint(100, 580)
#
#gap = np.arange(start, stop)
#
#the_ts[gap] = np.nan
#
start = np.random.randint(100, 699)

stop = start + np.random.randint(300, 580)

#print(start, stop)

gap = np.arange(start, stop)

the_ts[gap] = the_ts[gap] + 10
#

tdt = datetime.datetime.today()

the_dates = [tdt - datetime.timedelta(days=x) for
             x in range(0, len(the_ts))]

#
#plt.figure()
#plt.plot(the_dates, the_ts)
#plt.show()

TS = TempStab(dates=the_dates, array=the_ts,
              breakpoint_method="bfast", deseason=True, max_num_periods=20)

RES = TS.analysis(homogenize=True)


[os.remove(files) for files in os.listdir(os.getcwd()) if files.endswith(".png")]

#BP = rdp_bp(TS.numdate, TS.prep)

#print(BP)

#BP = rdp_bp_iter(TS.numdate, TS.prep, nIter=25, tol=0.2)

#print(BP)

#plt.figure()
#plt.plot(abs(spline_red_array_d2) == min(abs(spline_red_array_d2)))
#plt.show()

i=0
while len(TS.breakpoints)>0:
    plt.figure(figsize=(25,15))
    plt.ylim(-55,15)
    plt.plot(TS.numdate, TS.array, label = "original")
    plt.plot(TS.numdate, TS.array - TS.__trend_removed__, label = "no trend")
    plt.plot(TS.numdate, TS.prep, label = "no trend, no season")
    plt.plot(TS.numdate, TS.__season_removed__, label = "season")
    plt.plot(TS.numdate, TS.__trend_removed__, label = "trend")
    plt.plot(TS.numdate, TS.__season_removed__ + TS.__trend_removed__, label = "trend + season")
    [plt.axvline(x=xi, color='k', linestyle='--') for xi in TS.numdate[TS.breakpoints].tolist()]
    plt.plot(TS.numdate, RES["yn"], label = "BP_corrected")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
#    plt.plot(BP['rn'], BP['ra'])
#    for bp in BP['bp']:
#        plt.axvline(x=bp, color='y', linestyle='--')
    plt.savefig('process_' + str(i) + '.png')
    i+=1
    oldbp=TS.breakpoints
    print("REANALYSIS started!")
    TS.reanalysis()
    BP = rdp_bp(TS.numdate, TS.prep)
    if (len(TS.breakpoints) == len(oldbp)) and all(TS.breakpoints == oldbp):
        break

plt.figure(figsize=(25,15))
plt.ylim(-55,15)
plt.plot(TS.numdate, TS.array, label = "original")
plt.plot(TS.numdate, TS.array - TS.__trend_removed__, label = "no trend")
plt.plot(TS.numdate, TS.prep, label = "no trend, no season")
plt.plot(TS.numdate, TS.__season_removed__, label = "season")
plt.plot(TS.numdate, TS.__trend_removed__, label = "trend")
plt.plot(TS.numdate, TS.__season_removed__ + TS.__trend_removed__, label = "trend + season")
#[plt.axvline(x=xi, color='k', linestyle='--') for xi in TS.numdate[TS.breakpoints].tolist()]
plt.plot(TS.numdate, RES["yn"], label = "BP_corrected")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
#plt.plot(BP['rn'], BP['ra'])
#for bp in BP['bp']:
#    plt.axvline(x=bp)
plt.savefig('process_last.png')

#print("REANALYSIS started!")
#TS.reanalysis()
plt.figure(figsize=(25,15))
plt.ylim(-55,15)
plt.plot(TS.numdate, TS.homogenized, label = "final")
plt.plot(TS.numdate, the_ts, label = "original")
plt.plot(TS.numdate, TS.__trend_removed__, label = "trend")
plt.plot(TS.numdate, TS.__season_removed__ + TS.__trend_removed__, label = "trend + season")
plt.legend()
plt.savefig('test.png')
#
#print(np.diff(BP["bp"]))
#print(np.array(seqs))

P=(signal.convolve(TS.__identified_gaps__,np.array([1,1,1])/3.)[1:-1]!=0)
