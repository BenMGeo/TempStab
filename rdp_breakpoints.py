#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:01:15 2017

@author: bmueller
"""

from TempStab import rdp
from scipy import interpolate
import numpy as np


def __calc_zero_lin__(xarr, yarr):
    m, b = np.polyfit(xarr, yarr, 1)
    zeropoint = (0-b)/m
    return zeropoint


def rdp_bp(x, y):

    points = [[x[i], y[i]] for i in range(len(y))]
    Res_points = rdp.rdp(points, np.std(y))
    red_array = np.array([RP[1] for RP in Res_points])
    red_numdate = np.array([RP[0] for RP in Res_points])

    if all(red_numdate[i] <= red_numdate[i+1] for i in xrange(len(red_numdate)-1)):
        print "data sorted"
        f = interpolate.splrep(red_numdate, red_array, s=np.std(y)*len(red_array))
    else:
        print "data unsorted"
        f = interpolate.splrep(red_numdate[::-1], red_array[::-1], s=np.std(y)*len(red_array))

    spline_red_array_d2 = interpolate.splev(x, f, der=2)

    points = [[x[i], spline_red_array_d2[i]] for i in range(len(y))]
    Res_points = rdp.rdp(points, np.std(spline_red_array_d2))
    red_array_d2 = np.array([RP[1] for RP in Res_points])
    red_numdate_d2 = np.array([RP[0] for RP in Res_points])


    break_points = [__calc_zero_lin__(red_numdate_d2[i:(i+2)], red_array_d2[i:(i+2)]) for i in range(len(red_numdate_d2)-1)]

    r = {}
    r.update({"bp": break_points})
    r.update({"ra": red_array})
    r.update({"rn": red_numdate})

    return r


def __subset_mean__(start_stop, x, y):
    take = np.logical_and(x >= min(start_stop), x <= max(start_stop))
    return np.mean(y[take])


def rdp_bp_iter(x, y, nIter=100, tol=0):

    xsubset = [min(x), np.mean(x), max(x)]

    while nIter>0:

        if len(xsubset)>2:

            means = [__subset_mean__([xsubset[t],xsubset[t+1]], x, y) for t in range(len(xsubset)-1)]
            diff_list=(np.diff(means) < tol).tolist()
            diff_list.insert(0, False)
            diff_list.insert(-1, False)
            print(diff_list)


        nIter -= 1

    pass



































