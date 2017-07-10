#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:19:40 2017

@author: bmueller
"""
import datetime
import math
from itertools import compress
import numpy as np
from models import LinearTrend, SineSeason1, SineSeason3
# import matplotlib.pyplot as plt


class TempStab(object):
    """
    module to calculate measures of temporal stability
    """

    def __init__(self, dates, array, **kwargs):
        """
        Parameters
        ----------
        array: numpy nd.array
        """

        self.dates = dates
        self.array = array
        self.prep = None
        self.season = []
        self.season_mod = self.__do_nothing__
        self.numdate = np.linspace(1, 100, num=len(dates))
        self.filler = None
        self.__season_removed__ = None
        self.__trend_removed__ = None
        self.__orig__ = array.copy()
        self.__prep_orig__ = array.copy()
        self.__numdate_orig__ = self.numdate.copy()
        self.__identified_gaps__ = []
        self.__min_time_step__ = None

        # constants:
        # numeric tolerance for shift
        self.num_tol = 10**-8
        self.frequency = 365.2425
        self.period = 1

        # Initialize methods
        self.break_method = kwargs.get('breakpoint_method', None)
        self.__default_season__ = kwargs.get('default_season',
                                             [1., 0., 0., 0., 0., 0.])
        self.__homogenize__ = kwargs.get('homogenize', None)
        self.__timescale__ = kwargs.get('timescale', 'months')

        self.__run__ = kwargs.get('run', False)

        # Set methods, conversion and integritiy checks
        self.__set_break_model__()
        self.__set_season_model__()
        self.__set_time__()
        self.__scale_time__()
        self.__check__()

        # Run routine
        self.__preprocessing__(**kwargs)

        if self.__run__:
            self.analysis(homogenize=self.__homogenize__, **kwargs)

    def __do_nothing__(self, **kwargs):
        """
        does nothing!
        """
        pass

    def set_num_tol(self, num_tol):
        """
        set the tolerance of numerical dates for finding gaps
        """
        self.num_tol = num_tol

    def set_frequency(self, frequency):
        """
        set the annual frequency of numerical dates for finding gaps
        """
        self.frequency = frequency

    def __check__(self):
        """
        checks the integrity of the input
        """
        assert self.break_method is not None, \
            'Method for breakpoint estimation needs to be provided'
        assert len(self.dates) == len(self.array), \
            'Timeseries and data need to have the same dimension'
        assert isinstance(self.dates[0], datetime.datetime), \
            'Dates are not of type datetime'

    def __set_break_model__(self):
        pass

    def __set_season_model__(self):
        # note that the seasonal model needs to be consistent with the one
        # used for the breakpoint estimation!!!
        # thus if BFAST is used, a similar seasonal model needs to be applied
        self.season_mod = SineSeason3
        # at the moment use the 3fold sine model like
        # in Verbesselt et al. (2010), Eq. 3/4

    def __set_time__(self):
        """
        convert datetime array into float array
        """

        def toordinaltime(this_datetime):
            """
            calculate ordinals plus time fraction
            """
            time = this_datetime.hour * 1. + \
                (this_datetime.minute * 1. +
                 (this_datetime.second * 1. +
                  this_datetime.microsecond/1000000.)/60.)/60.

            days = this_datetime.toordinal() + time/24.

            return days

        self.numdate = np.array([toordinaltime(d) for d in self.dates])

        self.__set_period__()

    def __set_period__(self):
        """
        determine periodic frequency of data sampling from data
        """
        # TODO: what is this supposed to do?
        self.period = 24

    def __scale_time__(self):
        """
        scale to years, months, days
        """
        # TODO:
        # * change self.numdate
        # * change self.frequency
        # * change self.period
        # * maybe change self.dates
        pass

    def __preprocessing__(self, **kwargs):
        """
        perform preprocessing of the actual data
        The following options are available

        detrend : bool
            perform a linear detrending of the original data
        remove_season : bool
            remove the seasonality according to the model chosen
            if this option is used, then the overall linear trend
            is removed first and then the seasonality is removed thereafter
        """
        detrend = kwargs.get('detrend', False)
        deseason = kwargs.get('deseason', False)

        self.prep = self.__orig__.copy()

        # in case that season shall be removed do detrending first
        if deseason:
            detrend = True

        #  remove linear trend res = x - (slope*t + offset)
        if detrend:
            self.__detrend__()

        # remove seasonality
        if deseason:
            self.__deseason__()
        else:
            self.__season_removed__ = np.zeros_like(self.numdate)

    def __deseason__(self):
        print('Deseasonalization of the data ...')
        keep = [not math.isnan(pp) for pp in self.prep]
        loc_prep = np.array(list(compress(self.prep, keep)))
        loc_numdate = np.array(list(compress(self.numdate, keep)))
        sins = SineSeason1(loc_numdate, loc_prep, f=self.frequency)
        sins.fit()  # estimate seasonal model parameters
        self.__season_removed__ = sins.eval_func(self.numdate)
        self.prep -= self.__season_removed__
        print('Deseasonalization finished.')

    def __detrend__(self):
        """
        substracting linear trend
        """
        print('Detrending of the data ...')
        keep = [not math.isnan(pp) for pp in self.prep]
        loc_prep = np.array(list(compress(self.prep, keep)))
        loc_numdate = np.array(list(compress(self.numdate, keep)))
        lint = LinearTrend(loc_numdate, loc_prep)
        lint.fit()
        self.__trend_removed__ = lint.eval_func(self.numdate)
        self.prep -= self.__trend_removed__
        print('Detrending finished.')

    def analysis(self, homogenize=None, **kwargs):

        self.__prep_orig__ = self.prep.copy()
        self.__numdate_orig__ = self.numdate.copy()

        assert homogenize is not None, \
            'Homogenization argument need to be explicitely provided'

        # fill data gaps if existing
        self.__identify_gaps__()
        self.__fill_gaps__()
#
#        # identify breakpoints in time based;
#        # returns an array of indices with breakpoints
#        self.breakpoints = self._calc_breakpoints(self.array, **kwargs)
#
#        print self.breakpoints
#        # print self.breakpoints, type(self.breakpoints), len(self.breakpoints)
#        if len(self.breakpoints) > 0:
#            # estimate linear trend parameters for
#            # each section between breakpoints
#            self.trend = self._get_trend_parameters(self.breakpoints, self.x)
#
#            # in case that breakpoints occur, perform a normalizaion of the
#            # timeseries which corresponds to a removal of the segmentwise
#            # linear offsets of the linear trends, the slope is not corrected
#            # for
#            if homogenize:
#                yn = self._homogenization()
#                # fig = plt.figure()
#                # ax = fig.add_subplot(111)
#                # ax.plot(self.x)
#                # ax.plot(yn)
#                # plt.show()
#                # assert False
#            else:
#                yn = self.array
#        else:
#            self.trend = None
#            yn = self.array
#
#        # perform final trend estimation
#        L = LinearTrend(self.dates, yn)  # TODO uncertatinties???
#        L.fit()  # should store also significance information if possible

        res = {}
        res.update({"array": self.array})
#        res.update({'trend': {'slope': L.param[0], 'offset': L.param[1]}})
#        res.update({'yn': yn*1.})
#        res.update({'yorg': self.x*1.})
#        res.update({'yraw': self._raw*1.})
#        res.update({'season': self._season_removed*1.})
#        res.update({'breakpoints': self.breakpoints})
#        res.update({'nbreak': len(self.breakpoints)})

        return res

    def __fill_gaps__(self):
        """
        fill data gaps if existing
        """
        if sum(self.__identified_gaps__) > 0:
            keep = [not math.isnan(pp) for pp in self.prep]
            loc_prep = np.array(list(compress(self.prep, keep)))
            loc_numdate = np.array(list(compress(self.numdate, keep)))
            self.filler = self.numdate[self.__identified_gaps__] * 0. -999.

            if self.__season_removed__ is not None:
                self.__linear_p_noise__()
            elif self.__trend_removed__ is not None:
                self.__linear_p_noise__()
                self.__gap_season__(loc_numdate, loc_prep)
            else:
                self.__linear_p_noise__()
                self.__gap_season__(loc_numdate, loc_prep)
                self.__gap_trend__(loc_numdate, loc_prep)

            self.prep[self.__identified_gaps__] = self.filler
        else:
            pass

    def __linear_p_noise__(self):
        """
        produces linear filler with noise
        """

        pass

    def __gap_season__(self, t, x):
        """
        produces linear filler with season (no noise)
        """
        pass
#        season = self.season_mod(t=t,
#                                 x=x,
#                                 f=self.frequency)
#        season.fit()
#        self.filler += season.eval_func(self.numdate[self.__identified_gaps__])

    def __gap_trend__(self, t, x):
        """
        produces linear filler with trend (no noise)
        """
        pass

    def __identify_gaps__(self):
        """
        identify data gaps in self.array
        """
        self.__min_time_step__ = self.__get_min_timestep__()
        self.__set_up_new_ts__()
        self.__fill_with_na__()
        self.__calculate_na_indices__()
        self.__identified_gaps__ = self.__calculate_na_indices__()

    def __get_min_timestep__(self):
        """
        calculates the minimum timestep
        """
        return (np.diff(self.numdate)).min()

    def __set_up_new_ts__(self):
        """
        set a new ts based on the minimum timestep
        """
        num = np.abs((self.numdate.max() - self.numdate.min()) /
                     self.__min_time_step__) + 1
        new_dates = np.linspace(self.numdate.min(),
                                self.numdate.max(),
                                num=np.ceil(num))

        if self.numdate[0] == min(self.numdate):
            self.numdate = new_dates
        else:
            self.numdate = new_dates[::-1]

    def __fill_with_na__(self):
        """
        fill any gap with nan
        """
        new_array = self.prep * np.NAN
        keep = (self.numdate - self.__numdate_orig__) < self.num_tol
        new_array[keep] = self.prep[keep]
        self.prep = new_array

    def __calculate_na_indices__(self):
        """
        get a boolean array as indices for nan values
        """
        gaps = np.isnan(self.prep)
        return gaps
