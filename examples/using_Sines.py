#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:09:22 2017

@author: bmueller
"""
import numpy as np
from TempStab import LinearTrend, SineSeasonk, SineSeason1, SineSeason3
import matplotlib.pyplot as plt

t = np.arange(400)
x = np.sin(t*2*np.pi/100+5)
plt.plot(t, x)

S = SineSeason1(t, x, f=10)
S.fit()
print(S.param)
fitted = S.eval_func(t)

plt.plot(t, x)
plt.plot(t, fitted)
