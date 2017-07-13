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

xs = np.linspace(1,500,500)
m = 0.02
t = 3
lin = m*xs+t

yearly = 4.8 * np.sin(np.pi*2*((xs+25)/364))
monthly = 3.1 * np.cos(np.pi*2*((xs+1)/30))
daily = 1.1 * np.sin(np.pi*2*((xs+8)/1))

noise = (np.random.rand(len(xs)) - 0.5) * 3

the_ts = lin + yearly + monthly + daily + noise


def autocorr(x):
    result = np.correlate(x, x, mode='same')
    return result[result.size/2:]

def autocorrelation(x):
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x-np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:x.size/2]/np.sum(xp**2)

plt.plot(autocorrelation(the_ts))
plt.show()
plt.plot(autocorr(the_ts))
plt.show()
plt.plot(autocorr(the_ts), autocorrelation(the_ts))
plt.show()

start = np.random.randint(100, 399)

stop = start + np.random.randint(10, 80)

gap = np.arange(start, stop)

the_ts[gap] = np.nan


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
