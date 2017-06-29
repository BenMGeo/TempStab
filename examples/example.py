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

the_ts = np.concatenate(
        (3 + np.random.rand(100),
         6 + np.random.rand(100),
         4 + np.random.rand(100))
        )
the_dates = [datetime.datetime.today() - datetime.timedelta(days=x, minutes=7.3345*x) for
             x in range(0, len(the_ts))]

plt.figure()
plt.plot(the_dates, the_ts)
plt.show()

TS = TempStab(dates=the_dates, array=the_ts, breakpoint_method="this")
