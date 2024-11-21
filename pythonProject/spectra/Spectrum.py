# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:12:53 2023

@author: NaMiLAB
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy
import scipy.signal as signal
import spectra.BaselineRemoval as BaselineRemoval
from spectra.mlesg import MLESG as MLESG


def findIndent(value, array):
    indent = 0
    if array.ndim > 1:
        raise ValueError("Must be an 1D array")

    differences = np.abs(array - value)
    indents = np.where(differences == np.min(differences))
    if len(indents[0]) > 1:
        print("More than one matching value in array.")

    indent = indents[0][0]

    return indent


class Spectrum:
    xData = []
    yData = []
    xDataLabel = ""
    yDataLabel = ""
    date = datetime.date(1, 1, 1)
    tag = ""

    def __init__(self, xData, yData, xDataLabel, yDataLabel, date, tag):
        self.xData = np.array(xData)
        self.yData = np.array(yData)
        self.xDataLabel = xDataLabel
        self.yDataLabel = yDataLabel
        if type(date) == datetime.date:
            self.date = date
        else:
            try:
                self.date = datetime.date(date[0], date[1], date[2])
            except ValueError:
                print("Invalid date.")
                self.date = datetime.date(1, 1, 1)
            except TypeError:
                print("Invalid date")
                self.date = datetime.date(1, 1, 1)
        self.tag = tag

    def plot(self, save=False, path='', fname=''):
        plt.figure()
        plt.plot(self.xData, self.yData)
        plt.xlabel(self.xDataLabel)
        plt.ylabel(self.yDataLabel)
        if save:
            plt.savefig(fname=path + '\\' + fname)
        return

    def crop(self, minValue=None, maxValue=None):
        if minValue == None or minValue < min(self.xData):
            minValue = min(self.xData)
        if maxValue == None or maxValue > max(self.xData):
            maxValue = max(self.xData)
        minValue_ind = np.where(self.xData <= minValue)[0][-1]
        maxValue_ind = np.where(self.xData <= maxValue)[0][-1]
        self.xData = self.xData[minValue_ind:maxValue_ind]
        self.yData = self.yData[minValue_ind:maxValue_ind]
        return self

    def pad(self, points, minValue=None, maxValue=None):
        if minValue == None: minValue = self.xData[0]
        if maxValue == None: maxValue = self.xData[-1]

        if (minValue >= self.xData[0]) or (maxValue <= self.xData[-1]):
            print('Padding unnecesarry - given values already exist')
            return self
        else:
            padding_values = np.zeros(points)
            if minValue < self.xData[0]:
                left_padding = np.linspace(minValue, self.xData[0] - 1, points)
                self.xData = np.concatenate((left_padding, self.xData))
                self.yData = np.concatenate((padding_values, self.yData))
            if maxValue > self.xData[-1]:
                right_padding = np.linspace(self.xData[-1] + 1, maxValue, points)
                self.xData = np.concatenate((self.xData, right_padding))
                self.yData = np.concatenate((self.yData, padding_values))
            return self

    def resample(self, xnew, interpolation='linear'):
        # interpolation: linear or cubic spline?
        # xnew - a numpy array of values to evaluate on
        # xnew[0] - minimum, xnew[-1] - maximum az x tengelyen
        if (xnew[0] > self.xData[0]) or (xnew[-1] < self.xData[-1]):
            self.crop(xnew[0], xnew[-1])
        if (xnew[0] < self.xData[0]) or (xnew[-1] > self.xData[-1]):
            self.pad(xnew[0], xnew[-1])
        ynew = np.zeros(len(xnew))
        try:
            ynew = np.interp(xnew, self.xData, self.yData)
        except (ValueError, RuntimeError) as err:
            print(err)
        self.yData = ynew
        self.xData = xnew
        return self

    def baselineCorrect(self, lambda_=100, porder=1, repitition=15):
        """
        Implementation of Zhang fit fot baseline removal
        Originals: https://github.com/zmzhang/airPLS/

        Parameters
        ----------
        lambda_ : int, optional
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z. The default is 100.
        porder : int, optional
            adaptive iteratively reweighted penalized least squares
            for baseline fitting. The default is 1.
        repitition : int, optional
            how many iterations to run. The default is 15.

        Returns
        -------
        self: Spetrum
            baseline-corrected spectrum.

        """

        base = BaselineRemoval(self.yData)

        yDataCorr = base.ZhangFit(lambda_=lambda_, porder=porder, repitition=repitition)
        self.yData = yDataCorr
        return self

    def scaleMinMax(self, minValue, maxValue):
        self.yData = (minValue +
                      (self.yData - min(self.yData)) / (max(self.yData) - min(self.yData)) *
                      (maxValue - minValue))
        return self

    def shift(self, axis, value):
        """
        Shifts the data on a given axis with a given value

        Parameters
        ----------
        axis : str
            The axis to be shifted.
        value : float
            The value to be shifted with.

        Returns
        -------
        Spectrum
            The shifted spectrum.
        """
        if axis == 'x':
            self.xData = self.xData + value
        elif axis == 'y':
            self.yData = self.yData + value
        else:
            raise ValueError("Invalid axis. Must be 'x' or 'y'")
        return self


class RamanSpectrum(Spectrum):
    analyte = ""
    concentration = 0.  # mM
    power = 0.  # mW
    intTime = 0  # s
    avg = 1
    peakLoc = []
    peakData = {}

    def __init__(self, xData, yData, date, analyte, concentration, power, inTime, avg, tag=''):
        Spectrum.__init__(self, xData, yData, "Raman-shift (1/cm)", "Intensity (counts)",
                          date, tag)
        self.analyte = analyte
        self.concentration = concentration
        self.power = power
        self.intTime = inTime
        self.avg = avg

    def filterSG(self, windowLenght, polyorder, deriv=0, delta=1.0, axis=-1,
                 mode='interp', cval=0.0):
        """
        Savitzky-Golay filter implemented in scipy.signal

        Parameters
        ----------
        windowLenght : int
            The length of the filter window (i.e., the number of coefficients).
            If mode is ‘interp’, window_length must be less than or equal to the size of x.
        polyorder : int
            The order of the polynomial used to fit the samples.
            polyorder must be less than window_length.
        deriv : int, optional
            The order of the derivative to compute.
            This must be a nonnegative integer. The default is 0,
            which means to filter the data without differentiating.
        delta : float, optional
            The spacing of the samples to which the filter will be applied.
            This is only used if deriv > 0. The default is 1.0.
        axis : int, optional
            The axis of the array x along which the filter is to be applied.
            The default is -1.
        mode : str, optional
            Must be ‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’.
            This determines the type of extension to use for the padded signal
            to which the filter is applied. When mode is ‘constant’,
            the padding value is given by cval. When the ‘interp’ mode
            is selected (the default), no extension is used.
            Instead, a degree polyorder polynomial is fit to the last
            window_length values of the edges,and this polynomial is used to
            evaluate the last window_length//2 output values.The default is 'interp'.
        cval : scalar, optional
            Value to fill past the edges of the input if mode is ‘constant’. The default is 0.0.

        Returns
        -------
        yDataCorr : array_like
            Filtered data.

        """
        yDataCorr = signal.savgol_filter(self.yData, windowLenght, polyorder,
                                         deriv=deriv, delta=delta, axis=axis,
                                         mode=mode, cval=cval)
        self.yData = yDataCorr
        return self

    def filterMLESG(self, plot=False, peaks=[]):
        """

        Parameters
        ----------
        plot : boolean, optional
            If True, plots the filtered spectrum. The default is False.
        peaks : list or ndarray, optional
            Known Raman peaks of the analyte (wavenumbers). The default is [].

        Returns
        -------
        filtered : ndarray
            Filtered data.

        """

        if len(peaks) > 0:
            filtered = MLESG(self.yData, self.xData, peaks)
        else:
            self.findPeaks()
            filtered = MLESG(self.yData, self.xData, self.peakLoc)
            self.yData = filtered
        if plot:
            plt.figure()
            plt.plot(self.xData, self.yData)
        return self

    def scale(self, refConc, refPow, refIntTime):
        self.yData = (self.yData * (refConc / self.concentration) * (refPow / self.power)
                      * (refIntTime / self.intTime))
        self.concentration = refConc
        self.power = refPow
        self.intTime = refIntTime
        return self

    def findPeaks(self, height=None, threshold=None, distance=None, prominence=None,
                  width=None, wlen=None, rel_height=0.5, plateau_size=None):
        """
        Finds all the peaks in the spectrum.

        Parameters
        ----------
        height : number or ndarray or sequence, optional
            Required height of peaks. Either a number, None, an array matching
            x or a 2-element sequence of the former. The first element is always
            interpreted as the minimal and the second, if supplied, as the maximal
            required height.
        threshold : number or ndarray or sequence, optional
            Required threshold of peaks, the vertical distance to its neighboring
            samples. Either a number, None, an array matching x or a 2-element
            sequence of the former. The first element is always interpreted as
            the minimal and the second, if supplied, as the maximal required threshold.
        distance : number, optional
            Required minimal horizontal distance (>= 1) in samples between
            neighbouring peaks. Smaller peaks are removed first until the condition
            is fulfilled for all remaining peaks.
        prominence : number or ndarray or sequence, optional
            Required prominence of peaks. Either a number, None, an array matching
            x or a 2-element sequence of the former. The first element is always
            interpreted as the minimal and the second, if supplied, as the maximal
            required prominence.
        width : number or ndarray or sequence, optional
            Required width of peaks in samples. Either a number, None, an array
            matching x or a 2-element sequence of the former. The first element
            is always interpreted as the minimal and the second, if supplied, as
            the maximal required width.
        wlen : int, optional
            Used for calculation of the peaks prominences, thus it is only used
            if one of the arguments prominence or width is given. See argument
            wlen in peak_prominences for a full description of its effects.
        rel_height : float, optional
            Used for calculation of the peaks width, thus it is only used if
            width is given. See argument rel_height in peak_widths for a full
            description of its effects.
        plateau_size : number or ndarray or sequence, optional
            Required size of the flat top of peaks in samples. Either a number,
            None, an array matching x or a 2-element sequence of the former. The
            first element is always interpreted as the minimal and the second,
            if supplied as the maximal required plateau size.

        Returns
        -------
        peakLoc : ndarray
            Indices of peaks in x that satisfy all given conditions.
        peakData : dict
            A dictionary containing properties of the returned peaks which were
            calculated as intermediate results during evaluation of the specified conditions:
                ‘peak_heights’
                 If height is given, the height of each peak in x.
                ‘left_thresholds’, ‘right_thresholds’
                If threshold is given, these keys contain a peaks vertical distance
                to its neighbouring samples.
                ‘prominences’, ‘right_bases’, ‘left_bases’
                If prominence is given, these keys are accessible.
                See peak_prominences for a description of their content.
                ‘width_heights’, ‘left_ips’, ‘right_ips’
                If width is given, these keys are accessible.
                See peak_widths for a description of their content.
                ‘plateau_sizes’, left_edges’, ‘right_edges’
                If plateau_size is given, these keys are accessible and
                contain the indices of a peak’s edges (edges are still part of
                the plateau) and the calculated plateau sizes.
        """

        peakLoc, peakData = signal.find_peaks(self.yData, height=height,
                                              threshold=threshold, distance=distance,
                                              prominence=prominence, width=width,
                                              wlen=wlen, rel_height=rel_height,
                                              plateau_size=plateau_size)
        self.peakData = peakData
        self.peakLoc = peakLoc
        return peakLoc, peakData

    def integratedInt(self, peakWl, peakRange=2, peakWidth=20, absoluteIntensity=False,
                      retPeakIndents=False, showIndents=False, prominence=0, height=0):
        """
        Calculates the intgrated intensity of a given peak. May also return the
        absolute intensity (height) of the peak, and its indices.

        Parameters
        ----------
        peakWl : float
            The raman shift/wavelenght of the peak for which the intensity is calculated.
        peakRange : int, optional
            The range in which a peak is considered identical to the nominal value
            given in peakWl. The default is 5.
        peakWidth: int, optional
            The minimal half-width of the peak. The default is 5.
        absoluteIntensity : boolean, optional
            If true, only the height of the peak is returned. The default is False.
        retPeakIndents : booelan, optional
            If true, only the indices of the selected peak are returned. The default is False.

        Raises
        ------
        ValueError
            If no peaks are present in the range (+/- peakRange) of the given wavelenght.

        Returns
        -------
        peakArea : float
            Area of the selected peak.
        peakHeight : float
            Height of the selected peak.
        peakIndents : list
            Indices of the selected peak, in order:
            index of the peak, its left base, its right base (in the whole domain)

        """

        # the intensity values need to be positive
        if not (np.less(self.yData, np.zeros(len(self.yData))).all()):
            minvalue = min(self.yData)
            self = self.shift('y', np.abs(minvalue))

        if prominence > 0:
            self.findPeaks(prominence=prominence)
        elif height > 0:
            self.findPeaks(height=height)
        else:
            self.findPeaks()
        # TODO add other versions of findPeaks
        selPeakIndex = findIndent(peakWl, self.xData)  # index a teljes tartományra vonatkozóan
        peakIndex = 0

        try:
            for i in range(len(self.peakLoc)):
                if (((selPeakIndex - peakRange) <= self.peakLoc[i]) and (
                        self.peakLoc[i] <= (selPeakIndex + peakRange))):
                    peakIndex = self.peakLoc[i]
                    peak_leftIndent = findIndent(self.xData[selPeakIndex] - peakWidth, self.xData)
                    peak_rightIndent = findIndent(self.xData[selPeakIndex] + peakWidth, self.xData)

                    # Tartomány szűkítése a csúcs környékére - a tartomány vége lesz legrosszabb esetben a base
                    local_xData = self.xData[peak_leftIndent:peak_rightIndent]
                    local_yData = self.yData[peak_leftIndent:peak_rightIndent]

                    local_peakData = signal.find_peaks(local_yData, prominence=max(local_yData) / 10)[1]
                    local_peakIndex = findIndent(max(local_peakData["prominences"]), local_peakData["prominences"])
                    local_leftBaseIndex = local_peakData["left_bases"][local_peakIndex]
                    local_rightBaseIndex = local_peakData["right_bases"][local_peakIndex]

                    leftBaseWl = local_xData[local_leftBaseIndex]
                    rightBaseWl = local_xData[local_rightBaseIndex]
                    leftBaseIndex = findIndent(leftBaseWl, self.xData)
                    rightBaseIndex = findIndent(rightBaseWl, self.xData)

                    if ('peak_heights' in self.peakData.keys()): peakHeight = self.peakData["peak_heights"][i]
                    break
            if (peakIndex == 0): raise ValueError("no peak found at given wavelenght")
        except ValueError:
            print("no peak found")
            leftBaseIndex = 0
            rightBaseIndex = 0
            peakArea = 0
            peakHeight = 0
        peakIndents = [peakIndex, leftBaseIndex, rightBaseIndex]
        peakArea = scipy.integrate.trapezoid(self.yData[leftBaseIndex:rightBaseIndex],
                                             self.xData[leftBaseIndex:rightBaseIndex])
        areaxtra = ((self.xData[rightBaseIndex] - self.xData[leftBaseIndex])
                    * min([self.yData[rightBaseIndex], self.yData[leftBaseIndex]])
                    + ((self.xData[rightBaseIndex] - self.xData[leftBaseIndex])
                       * (max([self.yData[rightBaseIndex], self.yData[leftBaseIndex]])
                          - min([self.yData[rightBaseIndex], self.yData[leftBaseIndex]]))))
        peakArea -= areaxtra

        if showIndents:
            plt.figure()
            plt.plot(self.xData, self.yData)
            plt.plot([self.xData[leftBaseIndex], self.xData[peakIndex], self.xData[rightBaseIndex]],
                     [self.yData[leftBaseIndex], self.yData[peakIndex], self.yData[rightBaseIndex]], 'ro')

        if absoluteIntensity: return peakHeight
        if retPeakIndents: return peakIndents
        return peakArea

    def calcEF(self, peakWl, refSpec, prominenceRef, peakRangeRef, prominenceSERS, peakRangeSERS, refPow=1, refConc=1,
               refIntTime=1):
        if type(refSpec) != type(self):
            raise TypeError("Expected Spectrum as refSpec, instead " + str(type(refSpec)) + " was given.")
        refInt = refSpec.integratedInt(peakWl, prominence=prominenceRef, peakRange=peakRangeRef)
        refInt = refInt * (refPow / refSpec.power) * (refConc / refSpec.concentration) * (refIntTime / refSpec.intTime)
        sersInt = self.integratedInt(peakWl, prominence=prominenceSERS, peakRange=peakRangeSERS)
        sersInt = sersInt * (refPow / self.power) * (refConc / self.concentration) * (refIntTime / self.intTime)
        EF = sersInt / refInt
        return EF


