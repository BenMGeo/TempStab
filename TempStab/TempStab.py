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
from scipy import signal
# from scipy.stats import iqr
import scipy.fftpack as fftpack
import scipy.optimize as optimize
from scipy.ndimage.filters import uniform_filter1d
import statsmodels.api as sm
# from sklearn.neighbors import KernelDensity
from models import LinearTrend, SineSeason3  # , SineSeasonk, SineSeason1
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

####
# Additional functions


def mysine(array, para1, para2, para3):
    """
    simple sine model
    """
    return para1 * np.sin(para2 * array + para3)


def multisine(array, para1, para2, para3):
    """
    multiple sine model
    """
    init = array*0.
    for i, _ in enumerate(para1):
        init += mysine(array, para1[i], para2[i], para3[i])
    return init


def wrapper_multisine(array, *args):
    """
    wrapper for multiple sine model
    """
    equal_len = int(1./3.*len(args))
    para1, para2, para3 = list(args[:equal_len]), \
        list(args[equal_len:2*equal_len]), \
        list(args[2*equal_len:3*equal_len])
    return multisine(array, para1, para2, para3)


# def mygauss(x, sigma):
#    """
#    simple gauss distribution model without defining mu
#    """
#    global mu
#    return mlab.normpdf(x, mu, sigma)


def nargmax(array, num=1):
    """
    get the n biggest values of x (np.array)
    """
    args = []
    array = array.astype(float)
    for _ in range(num):
        argmax = array.argmax()
        args.append(argmax)
        array[argmax] = -np.inf
    return args
####


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

        self.dates = dates[:]
        self.array = array.copy()
        self.prep = array.copy()
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
        self.frequency = 365.2425
        self.periods = np.array([1])

        # Initialize methods
        self.break_method = kwargs.get('breakpoint_method', None)
        self.__default_season__ = kwargs.get('default_season',
                                             [1., 0., 0., 0., 0., 0.])
        self.__homogenize__ = kwargs.get('homogenize', None)
        self.__timescale__ = kwargs.get('timescale', 'months')
        self.__num_periods__ = kwargs.get('num_periods', 3)
        # TODO smoothing filter size is a hard question
        self.smoothing = kwargs.get('smoothing4periods', 21)

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

    def set_frequency(self, frequency):
        """
        set the annual frequency of numerical dates for finding gaps
        """
        self.frequency = frequency

    def set_smoothing4periods(self, smoothing_window):
        """
        set smoothing window for simpler evaluation of periods (denoise)
        """
        self.smoothing = smoothing_window

    def __check__(self):
        """
        checks the integrity of the input
        """
#        assert self.break_method is not None, \
#            'Method for breakpoint estimation needs to be provided'
        assert len(self.dates) == len(self.array), \
            'Timeseries and data need to have the same dimension'
        assert isinstance(self.dates[0], datetime.datetime), \
            'Dates are not of type datetime'

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

        self.__set_periods__()

    def __set_periods__(self):
        """
        determine periodic frequencies of data sampling from data
        """
        print('Calculating the periods of the data ...')
        self.__detrend__()
        periods = []

        # normalization/standardization
        # probably unneccessary
#        self.prep = (self.prep - self.prep.mean())/self.prep.std()

        keep = [not math.isnan(pp) for pp in self.prep]
        loc_prep = np.array(list(compress(self.prep, keep)))
        loc_numdate = np.array(list(compress(self.numdate, keep)))

        # presmoothing needed
        loc_prep = uniform_filter1d(loc_prep, size=self.smoothing)
        self.prep[np.array(keep)] = loc_prep

        # usually, within the range of 25-30 repetitions,
        # the following tries result in an error
        while len(periods) < self.__num_periods__:

            try:
                prephat = fftpack.rfft(loc_prep)
                idx = (prephat**2).argmax()

                freqs = fftpack.rfftfreq(prephat.size,
                                         d=np.min(np.abs(
                                                 np.diff(loc_numdate)
                                                 ))/(2*np.pi))
                frequency = freqs[idx]

                amplitude = loc_prep.max()
                guess = [amplitude, frequency, 0.]

                keep = [not math.isnan(pp) for pp in self.prep]
                loc_prep = np.array(list(compress(self.prep, keep)))
                loc_numdate = np.array(list(compress(self.numdate, keep)))

                (amplitude, frequency, phase), pcov = optimize.curve_fit(
                    mysine, loc_numdate, loc_prep, guess)

                period = 2*np.pi/frequency

                this_sine = mysine(self.numdate, amplitude, frequency, phase)
                self.prep -= this_sine

                periods.append(period)
                # reoccurences much longer than the time series
                # don't make sense
                keep = np.abs(periods) < len(self.prep)
                periods = list(compress(periods, keep))

            except RuntimeError:
                # print(str(i) + " out of 100 frequencies calculated!")
                break

#########
#        # the histogram of the data (can be deleted)
#        n, bins, patches = plt.hist(periods, 100, normed=1,
#                                    facecolor='black', alpha=0.75)
#
#        plt.xlabel('Periods')
#        plt.ylabel('Probability')
#        plt.xlim([0, 500])
#        plt.grid(True)
#        plt.show()
#########

#        # kernel density for a smoother histogram
#        # bandwidth: http://www.stat.washington.edu/courses/
#        #                  stat527/s14/readings/Turlach.pdf
#        # (4a)
#        if len(periods) > 1:
#            bw = 1.06 * min([np.std(periods),
#                             iqr(periods)/1.34]) * (len(periods)**(-0.2))
#        else:
#            bw = 0.1  # default value;
#        # makes sense with only 1 period available
#
#        kde = KernelDensity(kernel='gaussian',
#                            bandwidth=bw, rtol=1E-4).\
#            fit(np.array(periods).reshape(-1, 1))
#
#        # smooth linspace for possible periods
#        # (just a bit more than those observed with a higher resolution)
#        temp_res = np.linspace(0.5*min(periods),
#                               1.5*max(periods),
#                               len(self.prep)*2)
#        kde_hist = np.exp(kde.score_samples(temp_res.reshape(-1, 1)))
#
#        plt.plot(temp_res, kde_hist)
#
#        # calculate peaks of smooth histogram
#        peaks = signal.argrelextrema(kde_hist, np.greater)[0]
#
#        # calculate highest peaks
#        # should actually come from kernels!
#        periods = [temp_res[peaks][i]
#                   for i in nargmax(np.array(kde_hist[peaks]),
#                                    self.__num_periods__)]

#########
#        this is supposed to find max values, but is not
#        print(temp_res[peaks])
#        print(temp_res[peaks][kde_hist[peaks].argmax()])
#
#        # trying to calculate and reduce the biggest peaks
#        # (not working properly)
#
#        # first guess for mu
#        mu = temp_res[peaks][kde_hist[peaks].argmax()]
#        # first guess for sigma
#        std_guess = [1]
#
#        # peak_range for optimization of gauss
#        # peak_range_width how large???
#        peak_range_width = 41.
#        peak_range = peaks[kde_hist[peaks].argmax()] +\
#            np.arange(-np.floor(peak_range_width/2.),
#                      +np.floor(peak_range_width/2.)+1)
#        peak_range = peak_range.astype(int)
#
#        # weights for peak_range opt
#        weights = mlab.normpdf(temp_res[peak_range],
#                               temp_res[peaks[kde_hist[peaks].argmax()]],
#                               0.1*peak_range_width)
#        weights = weights/weights.sum()
#        plt.plot(temp_res[peak_range], weights)
#
#        # optimization of a gaussian curve within the peak_range
#        (std), pcov = optimize.curve_fit(mygauss,
#                                         temp_res[peak_range],
#                                         kde_hist[peak_range],
#                                         std_guess,
#                                         sigma=weights,
#                                         absolute_sigma=True)
#
#        plt.plot(temp_res[peak_range], kde_hist[peak_range])
#        print(mu, std[0])
#        plt.plot(temp_res[peaks], kde_hist[peaks], '+')
#        plt.plot(temp_res, mygauss(temp_res, std[0]))
#
#        # substract peak from kde
#        kde_hist -= mygauss(temp_res, std[0])
#        plt.plot(temp_res, kde_hist)
#
#        # calculate new extremes
#        peaks = signal.argrelextrema(kde_hist, np.greater)[0]
#        plt.plot(temp_res[peaks], kde_hist[peaks], 'o')
#
#        print(temp_res[peaks])
#        print(temp_res[peaks][kde_hist[peaks].argmax()])
#########

        self.periods = periods
        print('Calculating periods finished.')

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

    def __deseason__(self):
        print('Deseasonalization of the data ...')
        keep = [not math.isnan(pp) for pp in self.prep]
        loc_prep = np.array(list(compress(self.prep, keep)))
        loc_numdate = np.array(list(compress(self.numdate, keep)))
#        sins = SineSeason1(loc_numdate, loc_prep, f=self.frequency)
#        sins.fit()  # estimate seasonal model parameters
#        self.__season_removed__ = sins.eval_func(self.numdate)
#        self.prep -= self.__season_removed__
        self.__season_removed__ = self.prep*0.
        self.__season_removed__[:] = 0.

        # presmoothing needed for better access on periods
        loc_prep = uniform_filter1d(loc_prep, size=self.smoothing)

        # setting best guess and bounds for seasons
        amplitudes = list(np.repeat((loc_prep.max()-loc_prep.min())/2,
                                    len(self.periods)))
        freqs = [2*np.pi/p for p in self.periods]
        guess = amplitudes + freqs + list(np.repeat(1., len(self.periods)))

        ubound = list(np.repeat(np.inf, len(self.periods))) + \
            [f*1.1 for f in freqs] + \
            list(np.repeat(np.inf, len(self.periods)))
        lbound = list(np.repeat(-np.inf, len(self.periods))) + \
            [f*0.9 for f in freqs] + \
            list(np.repeat(-np.inf, len(self.periods)))

        # fitting the curves to periods
        params, pcov = optimize.curve_fit(wrapper_multisine,
                                          loc_numdate,
                                          loc_prep,
                                          guess,
                                          bounds=(lbound, ubound))

        # updating periods
        self.periods = [2*np.pi/p for p in
                        list(params[len(self.periods):2*len(self.periods)])]

        self.__season_removed__ += wrapper_multisine(self.numdate,
                                                     *params)
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

        # identify breakpoints in time based;
        # returns an array of indices with breakpoints
        self.breakpoints = self.__calc_breakpoints__(self.array, **kwargs)

        print self.breakpoints
#        print self.breakpoints, type(self.breakpoints), len(self.breakpoints)
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
            self.filler = self.numdate * 0. - 999.

            if self.__season_removed__ is not None:
                self.__linear__()
            elif self.__trend_removed__ is not None:
                self.__linear__()
                self.__gap_season__(loc_numdate, loc_prep)
            else:
                self.__linear__()
                self.__gap_season__(loc_numdate, loc_prep)
                self.__gap_trend__(loc_numdate, loc_prep)

            self.prep[self.__identified_gaps__] = \
                self.filler[self.__identified_gaps__]
        else:
            pass

    def __linear__(self):
        """
        produces linear filler without noise
        """
        self.filler = self.filler * 0. + np.mean(self.prep[np.logical_not(
            self.__identified_gaps__)])

    def __gap_season__(self, time, array):
        """
        produces linear filler with season (no noise)
        """
        season = self.season_mod(t=time,
                                 x=array,
                                 f=self.frequency)
        season.fit()
        self.filler = season.eval_func(self.numdate)

    def __gap_trend__(self, time, array):
        """
        produces linear filler with trend (no noise)
        """
        # calculate gaps environments (starts and stops)
        enlarged_gaps = signal.convolve(self.__identified_gaps__,
                                        np.array([1, 1, 1]))[1:-1] != 0
        starts_n_stops = np.logical_xor(enlarged_gaps,
                                        self.__identified_gaps__)
        sns_pos = self.numdate[starts_n_stops]

        # full linear model
        # (get that from detrend? externalize a single function?)
        lint = LinearTrend(time, array)
        lint.fit()

        for i in range(len(sns_pos)/2):
            if sns_pos[(i*2)] <= sns_pos[(i*2+1)]:
                this_gap = np.logical_and(self.numdate > sns_pos[(i*2)],
                                          self.numdate < sns_pos[(i*2+1)])
            else:
                this_gap = np.logical_and(self.numdate > sns_pos[(i*2+1)],
                                          self.numdate < sns_pos[(i*2)])
            self.filler[this_gap] = \
                lint.eval_func(self.numdate[this_gap])

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
        keep = np.isclose(self.numdate, self.__numdate_orig__)
        new_array[keep] = self.prep[keep]
        self.prep = new_array

    def __calculate_na_indices__(self):
        """
        get a boolean array as indices for nan values
        """
        gaps = np.isnan(self.prep)
        return gaps

    def __calc_breakpoints__(self, x, **kwargs):
        """
        calculating breakpoints based on set function self.Break
        """
        return self.__chosen_break__(x,
                                     start=self.dates[0],
                                     frequency=self.periods,
                                     **kwargs)

    def __set_break_model__(self):
        """
        set break_method
        """

        if self.break_method is None:
            print("No breakpoint method assigned. Just gaps are filled.")
            self.__chosen_break__ = self.__break_none__
        elif self.break_method == 'olssum':
            self.__chosen_break__ = self.__break_olssum__
        elif self.break_method == 'bfast':
            self.__chosen_break__ = self.__break_bfast__
        elif self.break_method == 'dummy':
            self.__chosen_break__ = self.__break_dummy__
        elif self.break_method == 'wang':
            self.__chosen_break__ = self.__break_wang__
        else:
            assert False, 'ERROR: Unknown breakpoint method'

    def __break_none__(self, x, **kwargs):
        """
        no breakpoint analysis
        Returns
        res : ndarray
            array with breakpoint indices
        """
        return np.array([])

    def __break_wang__(self, x, **kwargs):
        assert False, \
            "Breakpoint method " + self.break_method + " not implemented yet!"

    def __break_dummy__(self, x, **kwargs):
        assert False, \
            "Breakpoint method " + self.break_method + " not implemented yet!"

    def __break_bfast__(self, x, **kwargs):
        assert False, \
            "Breakpoint method " + self.break_method + " not implemented yet!"

    def __break_olssum__(self, x, **kwargs):
        # remove overall linear trend using OLS
        T = sm.add_constant(self.t)
        model = sm.OLS(x,T)
        results = model.fit()
        #~ print results.params
        #~ r = breaks_cusumolsresid(results.resid)  # todo: what to set as ddof???

        # estimate potential breakpoints from residual timeseries
        r =  self._get_breakpoints_spline(self.t, results.resid)
        #~ print 'Breakpoints found: ', r
        return r
        #~ print r
        #~ assert False