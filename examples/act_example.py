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

    
XXX=np.genfromtxt('/media/bmueller/Work/BEN_Auslesen_Punkt/tas_POINTS-iBAV-05_observations_historical_NA_LMU-INTERPOL_REGNIE-v3-IDW-BILINEARcombine_3h_19810101-20141231.csv', delimiter=';', skip_header=10)


the_ts = XXX[:,4]
#
win=8

the_ts = np.array([np.sum(the_ts[x:(x+win-1)]) for x in np.arange(len(the_ts)) if not(x%win)])

tdt = datetime.datetime.today()

#the_dates = [tdt - datetime.timedelta(hours=x*3) for
#             x in range(0, len(the_ts))]
#the_dates = list(reversed(the_dates))

the_dates = [tdt - datetime.timedelta(days=x) for
             x in range(0, len(the_ts))]
the_dates = list(reversed(the_dates))

#
#plt.figure()
#plt.plot(the_dates, the_ts)
#plt.show()

TS = TempStab(dates=the_dates, array=the_ts,
              breakpoint_method="olssum",periods_method="autocorr",
              deseason=True, max_num_periods=3)

RES = TS.analysis(homogenize=True)


[os.remove(files) for files in os.listdir(os.getcwd()) if files.endswith(".png")]

#BP = rdp_bp(TS.numdate, TS.prep)
#
#BP = rdp_bp_iter(TS.numdate, TS.array, nIter=25, tol=0.2)

#plt.figure()
#plt.plot(abs(spline_red_array_d2) == min(abs(spline_red_array_d2)))
#plt.show()

print(TS.periods)

#[XXX[i*8] for i in TS.breakpoints]
i=0
while len(TS.breakpoints)>0:
    plt.figure(figsize=(25,15))
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
    #plt.plot(BP['rn'], BP['ra'])
    #for bp in BP['bp']:
    #    plt.axvline(x=bp)signal.
    plt.savefig('process_' + str(i) + '.png')
    i+=1
    oldbp=TS.breakpoints
    print("REANALYSIS started!")
    TS.reanalysis()
    if (len(TS.breakpoints) == len(oldbp)) and all(TS.breakpoints == oldbp):
        break

plt.figure(figsize=(25,15))
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
