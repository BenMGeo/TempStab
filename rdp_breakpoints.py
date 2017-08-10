#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:01:15 2017

@author: bmueller
"""

from TempStab import rdp
from scipy import interpolate
import numpy as np
from itertools import compress


def __calc_zero_lin__(xarr, yarr):
    m, b = np.polyfit(xarr, yarr, 1)
    zeropoint = (0-b)/m
    return zeropoint


def rdp_bp(x, y):

    points = [[x[i], y[i]] for i in range(len(y))]
    Res_points = rdp.rdp(points, np.std(y))
    red_array = np.array([RP[1] for RP in Res_points])
    red_numdate = np.array([RP[0] for RP in Res_points])

    if len(x) <=3:
        k=1
    else:
        k=3

    try:
        if all(red_numdate[i] <= red_numdate[i+1] for i in xrange(len(red_numdate)-1)):
            f = interpolate.splrep(red_numdate, red_array, s=np.std(y)*len(red_array), k=k)
        else:
            f = interpolate.splrep(red_numdate[::-1], red_array[::-1], s=np.std(y)*len(red_array), k=k)

    except:
            if len(x)>0:
                r = {}
                r.update({"bp": [min(x), max(x)]})
                r.update({"ra": y})
                r.update({"rn": x})
                return r
            else:
                r = {}
                r.update({"bp": []})
                r.update({"ra": y})
                r.update({"rn": x})
                return r

    spline_red_array_d2 = interpolate.splev(x, f, der=k-1)

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


def __subset_meany__(start_stop, x, y):
    return np.mean(__subset__(start_stop, x, y)["y"])


def __subset_biasy__(start_stop, x, y):
    try:
        m, b = np.polyfit(__subset__(start_stop, x, y)["x"], __subset__(start_stop, x, y)["y"], 1)
    except:
        b = 0
    return b


def __subset__(start_stop, x, y):
    take = np.logical_and(x >= min(start_stop), x <= max(start_stop))
    return {"x": x[take], "y": y[take]}


def rdp_bp_iter(x, y, nIter=100, tol=0):

    xsubset = [min(x), max(x)]
#    xsubset = (np.arange(min(x), max(x), ((max(x)-min(x))/10))).tolist()
#    xsubset.append(max(x))
#    xsubset = x[0::5].tolist()

    while nIter > 0:

        for i in range(len(xsubset)-1):

            subset = __subset__([xsubset[i], xsubset[i+1]], x, y)
            bp = rdp_bp(subset["x"], subset["y"])

            for bp_i in bp["bp"]:
                if (bp_i >= min(xsubset) and bp_i <= max(xsubset)):
                    xsubset.append(bp_i)

        xsubset.sort()

        if len(xsubset) > 2:

            checks = [__subset_biasy__([xsubset[t], xsubset[t+1]], x, y) for t in range(len(xsubset)-1)]
            print(checks)
            keep_list = np.logical_not(np.diff(checks) < tol)
            keep_list = keep_list.tolist()
            keep_list.insert(0, True)
            keep_list.append(True)
            xsubset = list(compress(xsubset, keep_list))

        nIter -= 1

    return {"bp": xsubset}



































