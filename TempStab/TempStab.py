#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:19:40 2017

@author: bmueller
"""
import datetime
import numpy as np
from models import LinearTrend, SineSeason1


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
        self.__orig__ = array.copy()

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
        pass

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

        self.frequency = 365.2425  # number of days per year

        self.period = self.__set_period__()

    def __set_period__(self):
        """
        determine periodic frequency of data sampling from data
        """
        # TODO: what is this supposed to do?
        return 42.

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
        pass

    def __deseason__(self):
        print('Deseasonalization of the data ...')
        sins = SineSeason1(self.numdate, self.prep, f=self.frequency)
        sins.fit()  # estimate seasonal model parameters
        self.__season_removed__ = sins.eval_func(self.numdate)
        self.prep = self.prep - self.__season_removed__
        print('Deseasonalization finished.')

    def __detrend__(self):
        """
        substracting linear trend
        """
        print('Detrending of the data ...')
        lint = LinearTrend(self.numdate, self.prep)
        lint.fit()
        self.prep -= self.numdate * lint.param[0] + lint.param[1]
        print('Detrending finished.')

    def analysis(self, homogenize=None, **kwargs):
        """
        perform analysis of the actual data
        """
        res = {}
        return res
