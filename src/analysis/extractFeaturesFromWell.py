from __future__ import division
import numpy, pandas
from scipy.signal import butter
from scipy import interpolate
import scipy
import csv
import click
import sys, os, re, pprint
from scipy.optimize import curve_fit
from scipy.fftpack import fft
from scipy.signal import butter, lfilter, find_peaks_cwt, detrend, periodogram, remez, iirfilter
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline
from src.utils import metadataExtractor, cxpPrinter
import collections


def gcamp_interpolate(gcamp, number_of_additional_timepoints):
    gcamp_len = len(gcamp)
    timelabels = range(0, gcamp_len)
    cs = scipy.interpolate.CubicSpline(timelabels, gcamp)
    timelabels_spline = numpy.arange(0, gcamp_len-1, 1/number_of_additional_timepoints)
    gcamp_spline = cs(timelabels_spline)
    return gcamp_spline


def gcamp_normalize(gcamp, gcamp_min, gcamp_max):
    # signal min already remove during extraction
    return numpy.asarray(gcamp) / (gcamp_max - gcamp_min)


def gcamp_fwhm(gcamp, window_length, peak_ind, original_gcamp_length):
    win_rise = peak_ind - window_length if peak_ind >= window_length else 0
    win_fall = peak_ind + window_length + 1 if peak_ind < len(gcamp) - window_length else len(gcamp)
    gcamp_windowed = gcamp[win_rise:win_fall]  # look for a minimum within the window
    # argrelextrema requires an *order* less than or equal to half the length of the input array
    if window_length > len(gcamp_windowed) / 2:
        min_ind = scipy.signal.argrelextrema(gcamp_windowed, numpy.less,
                                             order=numpy.floor(len(gcamp_windowed) / 2).astype(int))
    else:
        min_ind = scipy.signal.argrelextrema(gcamp_windowed, numpy.less, order=window_length)
    if len(min_ind[0]) == 0:
        min_ind = numpy.where(gcamp_windowed == numpy.min(gcamp_windowed))

    fwhm_cutoff = (gcamp[peak_ind] - numpy.min(gcamp_windowed[min_ind])) / 2 + numpy.min(gcamp_windowed[min_ind])
    window_length_expanded = window_length * 2  # after determining a cutoff expand the search in case of assymettry between rise and fall
    # a fold change of 2 implies the decay of a signal could take twice as long as the activation of length *window_length*
    # alternatively, the entire time-series could be searched. This might be better since processing costs for this length of signal are neglgible
    win_rise_expanded = peak_ind - window_length_expanded if peak_ind >= window_length_expanded else 0
    win_fall_expanded = peak_ind + window_length_expanded + 1 if peak_ind < len(
        gcamp) - window_length_expanded else len(gcamp)
    gcamp_windowed_expanded = gcamp[win_rise_expanded:win_fall_expanded]
    peak_ind_expanded = peak_ind - win_rise_expanded
    # There are special cases when the signal in the window does not reach the *fwhm_cutoff*.
    # When this happens the fwhm will just use the ends of the window.
    # The first point past the cutoff is chosen by numpy.min() and numpy.max().
    # To choose the closest index, the first point just before the closet index must also be considered.
    fwhm_rise_ind = numpy.where(gcamp_windowed_expanded[:peak_ind_expanded] < fwhm_cutoff)
    if len(fwhm_rise_ind[0]) == 0:
        fwhm_rise = peak_ind - win_rise_expanded
    else:
        fwhm_riseA = numpy.asscalar(peak_ind_expanded - numpy.max(fwhm_rise_ind))
        fwhm_rise_testA = abs(gcamp_windowed_expanded[peak_ind_expanded - fwhm_riseA] - fwhm_cutoff)
        fwhm_rise_testB = abs(gcamp_windowed_expanded[peak_ind_expanded - fwhm_riseA + 1] - fwhm_cutoff)
        fwhm_rise = fwhm_riseA if fwhm_rise_testA <= fwhm_rise_testB else fwhm_riseA - 1
    fwhm_fall_ind = numpy.where(gcamp_windowed_expanded[peak_ind_expanded:] < fwhm_cutoff)
    if len(fwhm_fall_ind[0]) == 0:
        fwhm_fall = win_fall_expanded - peak_ind - 1  # the *-1* is to correct for an offset
    else:
        fwhm_fallA = numpy.asscalar(numpy.min(fwhm_fall_ind))
        fwhm_fall_testA = abs(gcamp_windowed_expanded[fwhm_fallA + peak_ind_expanded] - fwhm_cutoff)
        fwhm_fall_testB = abs(gcamp_windowed_expanded[fwhm_fallA + peak_ind_expanded - 1] - fwhm_cutoff)
        fwhm_fall = fwhm_fallA if fwhm_fall_testA <= fwhm_fall_testB else fwhm_fallA - 1
    # fwhm_rise and fwhm_fall should be greater than zero
    fwhm_rise = 1 if fwhm_rise == 0 else fwhm_rise
    fwhm_fall = 1 if fwhm_fall == 0 else fwhm_fall

    # peak width
    peak_start_ind = (peak_ind - fwhm_rise) if (peak_ind - fwhm_rise) > 0 else 0
    peak_end_ind = (peak_ind + fwhm_fall) if (peak_ind + fwhm_fall) < len(gcamp) else len(gcamp)-1
    peak_width = peak_end_ind - peak_start_ind  # same as fwhm_rise + fwhm_fall

    # area under the curve (area under the peak only)
    area_under_curve = numpy.trapz(gcamp[peak_start_ind:peak_end_ind+1], dx=original_gcamp_length/len(gcamp))
    return fwhm_rise, fwhm_fall, fwhm_cutoff, peak_width, area_under_curve


# To find in array the element closest to value
def find_nearest(array,value,startIdx,endIdx):
    if endIdx < len(array)-1:
        endIdx = endIdx+1
    idx = (numpy.abs(array[startIdx:endIdx]-value)).argmin() + startIdx
    return idx


# - To obtain half maximum points, peak start/end, height
# - Half max data not used currently, this method also returns other important
#   metrics such as peak height, etc.
def getPeakDefiningPoints(signal, peaks, valleys, wellmin):
    half_maximums, peak_halfmax_starts, peak_halfmax_ends  = [],[],[]  # halfmax values (halfmax,halfmax start, halfmax end)
    peak_rise_starts, peak_fall_ends= [],[]
    peak_heights_localmin, peak_heights_signalmin, peak_heights_wellmin = [],[],[]
    for idx,peak in enumerate(peaks):
        # Step 1: Get valleys between previous and current peak
        if len(peaks) > 1 and idx > 0:
            valleys_considered = valleys[(valleys > peaks[idx - 1]) & (valleys < peak)]
        else:
            valleys_considered = valleys[(valleys < peak)]

        # Step 2: Determine peak start index
        if len(valleys_considered) > 0:
            peak_start = valleys_considered[-1]  # 1st valley to the left of current peak
        else:
            peak_start = 0
        peak_rise_starts.append(peak_start)

        # Step 3: Determine peak end idx
        if idx <= len(peaks) - 2:  # if there is at least 1 more peak in peaks
            # valleys between current and next peak
            nextValleys = valleys[(valleys > peak) & (valleys < peaks[idx + 1])]
        else:
            # valleys between current peak and end of signal
            nextValleys = valleys[(valleys > peak) & (valleys < (len(signal)-1))]

        # take 1st valley to the right of current peak
        if len(nextValleys) > 0:
            peak_end = nextValleys[0]
        else:
            peak_end = len(signal) - 1
        peak_fall_ends.append(peak_end)

        # Step 4: Compute halfmax and approximate corresponding halfmax start/end index
        halfmax = (max(signal[peak] - signal[peak_start], signal[peak] - signal[peak_end]))/2.0 + signal[peak_start]
        half_maximums.append(halfmax)
        halfmax_start = find_nearest(signal, halfmax, peak_start, peak)
        peak_halfmax_starts.append(halfmax_start)
        peak_halfmax_ends.append(find_nearest(signal, signal[halfmax_start], peak, peak_end))

        # Step 5: Compute peak height
        # Method 1: Difference between gcamp signal and minimum value of that same gcamp signal.
        peakheight_signalmin = signal[peak] - min(signal)
        peak_heights_signalmin.append(peakheight_signalmin)

        # Method 2: Difference between gcamp signal and local minimum of the peak under analysis.
        peakheight_localmin = max(signal[peak] - signal[peak_start], signal[peak] - signal[peak_end])
        peak_heights_localmin.append(peakheight_localmin)

        # Method 3: Difference between gcamp signal and minimum gcamp value (avg background intensity of well)
        #           This difference correspond to the height of the signal itself as it is corrected for background intensity already.
        peakheight_wellmin = signal[peak]
        peak_heights_wellmin.append(peakheight_wellmin)

    return half_maximums, peak_halfmax_starts, peak_halfmax_ends, peak_rise_starts, peak_fall_ends, peak_heights_signalmin, peak_heights_localmin, peak_heights_wellmin


def wavelet_peak(gcamp, max_scale, min_length_0, min_snr_0, noise_perc_0):
    widths = numpy.arange(1,max_scale,1)
    peakind = find_peaks_cwt(detrend(gcamp), widths, max_distances=widths/2, gap_thresh=3, min_length=min_length_0, min_snr=min_snr_0, noise_perc=noise_perc_0)
    if len(peakind) == 0:
        peakind = [0]
    return peakind


"""
x:                    signal
min_peak_height:      anything smaller than that will be rejected
edge:                 {'rising','falling','both'} --> determine which indices to keep for irregular peaks, plateaus, etc.
valley:               if true, will returns indices of valleys instead of peaks
min_rel_height_neighbor:   specifies a minimum relative height difference between peaks and their immediate neighbors
min_peak_distance:    minimum distance that must separate each peak for them to be valid
keep_peaks_same_height: keep peaks of same height even if closer than min_peak_distance
Returns indices of identified peaks
"""
def find_peaks(x, min_peak_height=None, edge='rising', valley=False, min_rel_height_neighbor=0, min_peak_distance=1,
               keep_peaks_same_height=False):
    # need at least 3 points to identify valid peaks
    if x.size < 3:
        return numpy.array([], dtype=int)

    # if looking for valleys, invert the signal and look for peaks
    if valley:
        x = -x

    # identify the different types of peaks
    dx = numpy.diff(x)
    singlePointPeaks, risingEdgePeaks, fallingEdgePeaks = numpy.array([[], [], []], dtype=int)
    if not edge:
        singlePointPeaks = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            risingEdgePeaks = numpy.where((numpy.hstack((dx, 0)) <= 0) & (numpy.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            fallingEdgePeaks = numpy.where((numpy.hstack((dx, 0)) < 0) & (numpy.hstack((0, dx)) >= 0))[0]
    ind = numpy.unique(numpy.hstack((singlePointPeaks, risingEdgePeaks, fallingEdgePeaks)))

    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]

    # keep only peaks > minimum peak height
    if ind.size and min_peak_height is not None:
        ind = ind[x[ind] >= min_peak_height]

    # remove peaks that are less than "neighbor_threshold" higher than their neighbors
    if ind.size and min_rel_height_neighbor > 0:
        dx_neighbors = numpy.min(numpy.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = numpy.delete(ind, numpy.where(dx_neighbors < min_rel_height_neighbor)[0])

    # identify peaks closer to one another than min_peak_distance
    if ind.size and min_peak_distance > 1:
        ind = ind[numpy.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = numpy.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - min_peak_distance) & (ind <= ind[i] + min_peak_distance) \
                              & (x[ind[i]] > x[ind] if keep_peaks_same_height else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = numpy.sort(ind[~idel])

    return ind


# Returns wavelet analysis and periodogram stored in ordered dictionary
def wavelet_periodogram_extraction(gcamp, original_gcamp_length):
    d = collections.OrderedDict()

    # correction factor to account for interpolation
    correction_factor = original_gcamp_length / len(gcamp)

    # Wavelet 8 (8 is better as a general peak identifier)
    window_length_wavelet8 = 15
    peak_ind_wavelet8 = wavelet_peak(gcamp, 8, 5, 2, 10)
    # peak_ind_wavelet8 = [i for i in peak_ind_wavelet8 if gcamp[i] >= threshold]

    if len(peak_ind_wavelet8) == 0 or (len(peak_ind_wavelet8) == 1 and peak_ind_wavelet8[0] == 0):
        d["wavelet8_peak_count"] = 0
        d["wavelet8_firing_rate"] = 0
    else:
        # full-width half-maximum computations
        fwhm_wavelet8 = [gcamp_fwhm(gcamp, window_length_wavelet8, pk, original_gcamp_length) for pk in
                         peak_ind_wavelet8]
        fwhm_wavelet8_arr = numpy.asarray(fwhm_wavelet8)
        fwhm_wavelet8_average = numpy.average(fwhm_wavelet8_arr, 0)
        fwhm_wavelet8_sum = numpy.sum(fwhm_wavelet8_arr, 0)  # used for total AUC

        # add features to dictionary
        d["wavelet8_peak_count"] = len(peak_ind_wavelet8)
        d["wavelet8_firing_rate"] = len(peak_ind_wavelet8) / original_gcamp_length
        d["wavelet8_amplitude"] = numpy.mean(gcamp[peak_ind_wavelet8])
        d["wavelet8_fwhm_rise_time"] = fwhm_wavelet8_average[0] * correction_factor
        d["wavelet8_fwhm_fall_time"] = fwhm_wavelet8_average[1] * correction_factor
        d["wavelet8_fwhm_cutoff"] = fwhm_wavelet8_average[2]
        d["wavelet8_fwhm_peak_width"] = fwhm_wavelet8_average[3] * correction_factor
        d["wavelet8_fwhm_area_under_curve"] = fwhm_wavelet8_average[4]
        d["wavelet8_fwhm_total_area_under_curve"] = fwhm_wavelet8_sum[4]
        d["wavelet8_fwhm_rise_fall_ratio_mean"] = numpy.mean(
            numpy.divide(fwhm_wavelet8_arr[:, 0], fwhm_wavelet8_arr[:, 1]))
        if len(peak_ind_wavelet8) > 1:
            d["wavelet8_spike_interval_mean"] = numpy.mean(numpy.diff(peak_ind_wavelet8) * correction_factor)
            d["wavelet8_spike_interval_var"] = numpy.var(numpy.diff(peak_ind_wavelet8) * correction_factor)

    # Wavelet 4 (4 is good for identifying peaks of smaller amplitude)
    window_length_wavelet4 = 9
    peak_ind_wavelet4 = wavelet_peak(gcamp, 4, 3, 1, 10)

    if len(peak_ind_wavelet4) == 0 or (len(peak_ind_wavelet4) == 1 and peak_ind_wavelet4[0] == 0):
        d["wavelet4_peak_count"] = 0
        d["wavelet4_firing_rate"] = 0
    else:
        # full-width half-maximum computations
        fwhm_wavelet4 = [gcamp_fwhm(gcamp, window_length_wavelet4, pk, original_gcamp_length) for pk in
                         peak_ind_wavelet4]
        fwhm_wavelet4_arr = numpy.asarray(fwhm_wavelet4)
        fwhm_wavelet4_average = numpy.average(fwhm_wavelet4_arr, 0)
        fwhm_wavelet4_sum = numpy.sum(fwhm_wavelet4_arr, 0)  # used for total AUC

        # add features to dictionary
        d["wavelet4_peak_count"] = len(peak_ind_wavelet4)
        d["wavelet4_firing_rate"] = len(peak_ind_wavelet4) / original_gcamp_length
        d["wavelet4_amplitude"] = numpy.mean(gcamp[peak_ind_wavelet4])
        d["wavelet4_fwhm_rise_time"] = fwhm_wavelet4_average[0] * correction_factor
        d["wavelet4_fwhm_fall_time"] = fwhm_wavelet4_average[1] * correction_factor
        d["wavelet4_fwhm_cutoff"] = fwhm_wavelet4_average[2]
        d["wavelet4_fwhm_peak_width"] = fwhm_wavelet4_average[3] * correction_factor
        d["wavelet4_fwhm_area_under_curve"] = fwhm_wavelet4_average[4]
        d["wavelet4_fwhm_total_area_under_curve"] = fwhm_wavelet4_sum[4]
        d["wavelet4_fwhm_rise_fall_ratio_mean"] = numpy.mean(
            numpy.divide(fwhm_wavelet4_arr[:, 0], fwhm_wavelet4_arr[:, 1]))
        if len(peak_ind_wavelet4) > 1:
            d["wavelet4_spike_interval_mean"] = numpy.mean(numpy.diff(peak_ind_wavelet4) * correction_factor)
            d["wavelet4_spike_interval_var"] = numpy.var(numpy.diff(peak_ind_wavelet4) * correction_factor)

    # Periodogram (Fourier Fast Transform, Power Spectral Density)
    # For a typical 45-timepoint series, we expect 89 distinct frequencies along with their weights (aka power)
    # The number of distinct frequency components can be computed using the following formula:
    #  numFreq = len(gcamp)/2 + 1
    # If gcamp is non interpolated, then --> numFreq = ((len(gcamp)-1)*num_additional_points)/2 + 1
    f, Pxx_den = scipy.signal.periodogram(scipy.signal.detrend(gcamp))
    for index, power in enumerate(Pxx_den):
        key = "periodogram_{0}".format(index)
        d[key] = power
    return d


# Gets gcamp features (processes a single object, i.e. a gcamp time series)
def gcamp_feature_extraction(gcamp,well,obj_number, thresholds, original_gcamp_length, platename, well_types_dict, wellmins):
    # retrieve well type
    well_type = 'unspecified'  # default well type
    for key in well_types_dict.keys():
        if well in well_types_dict[key]:
            well_type = key
            break

    # instantiate ordered dictionary
    d = collections.OrderedDict()
    d["plate"] = platename
    d["well"] = well
    d["well_type"] = well_type
    d["obj_number"] = obj_number

    # signal stats
    d["signal_mean"] = numpy.mean(gcamp)
    d["signal_var"] = numpy.var(gcamp)
    d["signal_max"] = numpy.max(gcamp)
    d["signal_min"] = numpy.min(gcamp)

    # correction factor to account for interpolation
    correction_factor = original_gcamp_length / len(gcamp)

    ## 3 methods for identifying peaks (differential, wavelet8, wavelet4)
    # Method 1: New Peaks (method based on difference array)
    # sensitive to noise, but can be controlled by setting min distance between peaks
    # and min relative height to neighbors.
    peaks_indices = find_peaks(gcamp)  # min_peak_height=threshold
    valleys_indices = find_peaks(gcamp, valley=True, edge='both')

    # list of peak dictionaries for storing individual peak details
    peak_dicts = []

    if len(peaks_indices) == 0:
        peak_height_methods = ["WM", "SM", "LM"]  # WM: well min, SM: signal min, LM: local min
        for m in peak_height_methods:
            d["{0}_peak_count".format(m)] = 0
            d["{0}_firing_rate".format(m)] = 0

        # NOT necessarily needed, if obj number is not peak_data file it means obj had no peak
        d_peak = collections.OrderedDict()
        d_peak["plate"] = platename
        d_peak["well"] = well
        d_peak["well_type"] = well_type
        d_peak["obj_number"] = obj_number
        d_peak["peak_count"] = 0
        peak_dicts.append(d_peak)
    else:
        halfMaximums, peak_halfmax_starts, peak_halfmax_ends, \
        peak_rise_starts, peak_fall_ends, \
        peak_heights_signalmin, peak_heights_localmin, peak_heights_wellmin \
            = getPeakDefiningPoints(gcamp, peaks_indices, valleys_indices, wellmins)

        # convert to numpy arrays
        peak_heights_signalmin = numpy.asarray(peak_heights_signalmin)
        peak_heights_localmin = numpy.asarray(peak_heights_localmin)
        peak_heights_wellmin = numpy.asarray(peak_heights_wellmin)
        peak_rise_starts = numpy.asarray(peak_rise_starts)
        peak_fall_ends = numpy.asarray(peak_fall_ends)
        peaks_indices = numpy.asarray(peaks_indices)


        # baselined signals: to ensure areas calculated are positive
        if min(gcamp) < 0:
            baselined_gcamp = numpy.asarray(gcamp) + abs(min(gcamp))  # shift signal UP to 0 and add wellmin
        else:
            baselined_gcamp = numpy.asarray(gcamp) - min(gcamp)  # shift signal DOWN to 0 and add well min

        # for each peak, create a dictionary with its features
        for j, peak_index in enumerate(peaks_indices):

            # instantiate ordered dictionary for storing individual peak data
            d_peak = collections.OrderedDict()

            # plate,well,obj  (same for all peaks)
            d_peak["plate"] = platename
            d_peak["well"] = well
            d_peak["well_type"] = well_type
            d_peak["obj_number"] = obj_number

            # peak info
            d_peak['peak_count'] = 1
            d_peak['peak_index'] = peak_index * correction_factor  # this is also the peak id as (plate,well,obj,peak) is unique
            d_peak['peak_rise_start_index'] = peak_rise_starts[j] * correction_factor
            d_peak['peak_fall_end_index'] = peak_fall_ends[j] * correction_factor
            # (the following is apparently the same for all 3 methods)
            d_peak["rise_time"] = (peak_index - peak_rise_starts[j]) * correction_factor
            d_peak["fall_time"] = (peak_fall_ends[j] - peak_index) * correction_factor
            d_peak["rise_fall_ratio"] = peak_rise_starts[j] / peak_fall_ends[j]
            d_peak["peak_width"] = (peak_fall_ends[j] - peak_rise_starts[j]) * correction_factor
            # TODO: AUC is the same for all methods, to be fixed
            d_peak["area_under_curve"] = numpy.trapz(baselined_gcamp, dx=original_gcamp_length / len(gcamp))

            # peak height (3 methods: WM,SM,LM)
            d_peak["WM_amplitude"] = peak_heights_wellmin[j]
            d_peak["SM_amplitude"] = peak_heights_signalmin[j]
            d_peak["LM_amplitude"] = peak_heights_localmin[j]

            # area under individual peak
            d_peak["WM_area_under_peak"] = numpy.trapz(baselined_gcamp[peak_rise_starts[j]:peak_fall_ends[j]+1], dx=original_gcamp_length/len(gcamp))
            d_peak['SM_area_under_peak'] = numpy.trapz(baselined_gcamp[peak_rise_starts[j]:peak_fall_ends[j]+1], dx=original_gcamp_length/len(gcamp))
            d_peak['LM_area_under_peak'] = numpy.trapz(baselined_gcamp[peak_rise_starts[j]:peak_fall_ends[j]+1] - min(baselined_gcamp[peak_rise_starts[j]:peak_fall_ends[j] + 1]), dx=original_gcamp_length / len(gcamp))

            # append to peak_dicts
            peak_dicts.append(d_peak)


        ## 3 methods for computing peak height and corresponding feature set (well min, signal min, local min)
        # Method A: Height relative to WELL MIN (WM)
        peaks_above_WM_threshold = peak_heights_wellmin > thresholds[0]
        peak_heights_WM = peak_heights_wellmin[peaks_above_WM_threshold]
        peak_rise_starts_WM = peak_rise_starts[peaks_above_WM_threshold]
        peak_fall_ends_WM = peak_fall_ends[peaks_above_WM_threshold]
        peaks_indices_WM = peaks_indices[peaks_above_WM_threshold]

        if len(peaks_indices_WM) == 0:
            d["WM_peak_count"] = 0
            d["WM_firing_rate"] = 0
        else:
            # add features to dictionary
            d["WM_peak_count"] = len(peaks_indices_WM)
            d["WM_firing_rate"] = len(peaks_indices_WM) / original_gcamp_length
            d["WM_amplitude"] = numpy.mean(peak_heights_WM)
            d["WM_rise_time"] = numpy.mean(peaks_indices_WM - peak_rise_starts_WM) * correction_factor
            d["WM_fall_time"] = numpy.mean(peak_fall_ends_WM - peaks_indices_WM) * correction_factor
            d["WM_rise_fall_ratio"] = numpy.mean(numpy.asarray(peak_rise_starts_WM) / peak_fall_ends_WM)
            d["WM_peak_width"] = numpy.mean(numpy.asarray(peak_fall_ends_WM) - peak_rise_starts_WM) * correction_factor

            if min(gcamp) < 0:
                baselined_gcamp = numpy.asarray(gcamp) + abs(min(gcamp))  # shift signal up to 0 and add wellmin
            else:
                baselined_gcamp = numpy.asarray(gcamp) - min(gcamp)  # shift signal down to 0 and add well min
            area_under_peaks = [
                numpy.trapz(baselined_gcamp[peak_rise_starts_WM[i]:peak_fall_ends_WM[i]+1], dx=original_gcamp_length/len(gcamp))
                for i in range(len(peaks_indices_WM))
            ]
            d["WM_area_under_peak"] = numpy.mean(area_under_peaks)
            d["WM_total_area_under_peaks"] = numpy.sum(area_under_peaks)
            d["WM_area_under_curve"] = numpy.trapz(baselined_gcamp, dx=original_gcamp_length/len(gcamp))


            d["WM_area_under_peak"] = numpy.mean(area_under_peaks)
            d["WM_total_area_under_peaks"] = numpy.sum(area_under_peaks)

            if len(peaks_indices_WM) > 1:
                d["WM_spike_interval_mean"] = numpy.mean(numpy.diff(peaks_indices_WM) * correction_factor)
                d["WM_spike_interval_var"] = numpy.var(numpy.diff(peaks_indices_WM) * correction_factor)


        # Method B: Height relative to SIGNAL MIN (SM)
        peaks_above_SM_threshold = peak_heights_signalmin > thresholds[1]
        peak_heights_SM = peak_heights_signalmin[peaks_above_SM_threshold]
        peak_rise_starts_SM = peak_rise_starts[peaks_above_SM_threshold]
        peak_fall_ends_SM = peak_fall_ends[peaks_above_SM_threshold]
        peaks_indices_SM = peaks_indices[peaks_above_SM_threshold]

        if len(peaks_indices_SM) == 0:
            d["SM_peak_count"] = 0
            d["SM_firing_rate"] = 0
        else:
            # add features to dictionary
            d["SM_peak_count"] = len(peaks_indices_SM)
            d["SM_firing_rate"] = len(peaks_indices_SM) / original_gcamp_length
            d["SM_amplitude"] = numpy.mean(peak_heights_SM)
            d["SM_rise_time"] = numpy.mean(peaks_indices_SM - peak_rise_starts_SM) * correction_factor
            d["SM_fall_time"] = numpy.mean(peak_fall_ends_SM - peaks_indices_SM) * correction_factor
            d["SM_rise_fall_ratio"] = numpy.mean(numpy.asarray(peak_rise_starts_SM) / peak_fall_ends_SM)
            d["SM_peak_width"] = numpy.mean(numpy.asarray(peak_fall_ends_SM) - peak_rise_starts_SM) * correction_factor

            # baseline gcamp at 0 to obtain only positive areas
            if min(gcamp) < 0:
                baselined_gcamp = numpy.asarray(gcamp) + abs(min(gcamp))  # shift signal up to 0
            else:
                baselined_gcamp = numpy.asarray(gcamp) - min(gcamp)  # shift signal down to 0

            area_under_peaks = [
                numpy.trapz(baselined_gcamp[peak_rise_starts_SM[i]:peak_fall_ends_SM[i]+1], dx=original_gcamp_length/len(gcamp))
                for i in range(len(peaks_indices_SM))
            ]
            d["SM_area_under_peak"] = numpy.mean(area_under_peaks)
            d["SM_total_area_under_peaks"] = numpy.sum(area_under_peaks)
            d["SM_area_under_curve"] = numpy.trapz(baselined_gcamp, dx=original_gcamp_length/len(gcamp))

            if len(peaks_indices_SM) > 1:
                d["SM_spike_interval_mean"] = numpy.mean(numpy.diff(peaks_indices_SM) * correction_factor)
                d["SM_spike_interval_var"] = numpy.var(numpy.diff(peaks_indices_SM) * correction_factor)


        # Method C: Height measured relative to peak local minimum
        peaks_above_LM_threshold = peak_heights_localmin > thresholds[2]
        peak_heights_LM = peak_heights_localmin[peaks_above_LM_threshold]
        peak_rise_starts_LM = peak_rise_starts[peaks_above_LM_threshold]
        peak_fall_ends_LM = peak_fall_ends[peaks_above_LM_threshold]
        peaks_indices_LM = peaks_indices[peaks_above_LM_threshold]

        if len(peaks_indices_LM) == 0:
            d["LM_peak_count"] = 0
            d["LM_firing_rate"] = 0
        else:
            # add features to dictionary
            d["LM_peak_count"] = len(peaks_indices_LM)
            d["LM_firing_rate"] = len(peaks_indices_LM) / original_gcamp_length
            d["LM_amplitude"] = numpy.mean(peak_heights_LM)
            d["LM_rise_time"] = numpy.mean(peaks_indices_LM - peak_rise_starts_LM) * correction_factor
            d["LM_fall_time"] = numpy.mean(peak_fall_ends_LM - peaks_indices_LM) * correction_factor
            d["LM_rise_fall_ratio"] = numpy.mean(numpy.asarray(peak_rise_starts_LM) / peak_fall_ends_LM)
            d["LM_peak_width"] = numpy.mean(
                numpy.asarray(peak_fall_ends_LM) - peak_rise_starts_LM) * correction_factor

            # baseline gcamp at 0 to obtain only positive areas
            if min(gcamp) < 0:
                baselined_gcamp = numpy.asarray(gcamp) + abs(min(gcamp))  # shift signal up to 0
            else:
                baselined_gcamp = numpy.asarray(gcamp) - min(gcamp)  # shift signal down to 0

            area_under_peaks = [
                numpy.trapz(baselined_gcamp[peak_rise_starts_LM[i]:peak_fall_ends_LM[i] + 1] - min(baselined_gcamp[peak_rise_starts_LM[i]:peak_fall_ends_LM[i] + 1]),
                            dx=original_gcamp_length / len(gcamp))
                for i in range(len(peaks_indices_LM))
            ]
            d["LM_area_under_peak"] = numpy.mean(area_under_peaks)
            d["LM_total_area_under_peaks"] = numpy.sum(area_under_peaks)
            d["LM_area_under_curve"] = numpy.trapz(baselined_gcamp, dx=original_gcamp_length / len(gcamp))

            if len(peaks_indices_LM) > 1:
                d["LM_spike_interval_mean"] = numpy.mean(numpy.diff(peaks_indices_LM) * correction_factor)
                d["LM_spike_interval_var"] = numpy.var(numpy.diff(peaks_indices_LM) * correction_factor)

    # add wavelet analysis and periodogram to dictionary
    d.update(wavelet_periodogram_extraction(gcamp, original_gcamp_length))
    return d, peak_dicts


# process entire well, all objects in that well
# ind+1 is used to match object number with ids in segments image
def process_well(gcamp_arr_norm, peak_thresholds_norm, gcamp_arr_raw, peak_thresholds_raw, original_gcamp_length, platename, well, well_types_dict, segmentsdir, wellmins, outputdir):
    correction_factor = original_gcamp_length / len(gcamp_arr_norm[0])

    features_norm = [gcamp_feature_extraction(gcamp, well, ind + 1, peak_thresholds_norm, original_gcamp_length, platename,well_types_dict, wellmins) for ind, gcamp in enumerate(gcamp_arr_norm)]
    gcamp_list_of_features_norm = [obj_data for obj_data, peak_data in features_norm]
    peak_data_norm = [peak_data for obj_data, peak_data in features_norm]

    features_raw = [gcamp_feature_extraction(gcamp, well, ind + 1, peak_thresholds_raw, original_gcamp_length, platename, well_types_dict, wellmins) for ind, gcamp in enumerate(gcamp_arr_raw)]
    gcamp_list_of_features_raw = [obj_data for obj_data, peak_data in features_raw]
    peak_data_raw = [peak_data for obj_data, peak_data in features_raw]

    # convert into dataframes
    df_gcamp_norm = pandas.DataFrame(gcamp_list_of_features_norm)
    df_gcamp_raw = pandas.DataFrame(gcamp_list_of_features_raw)
    
    list_df_peak_data_norm = [pandas.DataFrame(cell) for cell in peak_data_norm]
    df_peak_data_norm = pandas.concat(list_df_peak_data_norm)

    list_df_peak_data_raw = [pandas.DataFrame(cell) for cell in peak_data_raw]
    df_peak_data_raw = pandas.concat(list_df_peak_data_raw)

    # create 'peak_data' dir if needed
    peakdir = 'peak_data'
    if not os.path.isdir(os.path.join(outputdir,peakdir)):
        os.makedirs(os.path.join(outputdir,peakdir))

    # save individual peak data to file (1 file per well)
    df_peak_data_norm.to_csv(os.path.join(outputdir,'peak_data', "{0}_peaks_norm.csv".format(well)), index=False)
    df_peak_data_raw.to_csv(os.path.join(outputdir,'peak_data', "{0}_peaks_raw.csv".format(well)), index=False)

    # adjust column ordering within the dataframes
    max_numcol_ind_norm = numpy.asscalar(numpy.argmax([len(d) for d in gcamp_list_of_features_norm]))
    df_gcamp_norm = df_gcamp_norm[gcamp_list_of_features_norm[max_numcol_ind_norm].keys()]
    max_numcol_ind_raw = numpy.asscalar(numpy.argmax([len(d) for d in gcamp_list_of_features_raw]))
    df_gcamp_raw = df_gcamp_raw[gcamp_list_of_features_raw[max_numcol_ind_raw].keys()]

    # drop duplicate columns and prepend 'RAW' to raw data columns
    # df_gcamp_raw = df_gcamp_raw.drop(['plate','well','well_type','obj_number'], axis=1)
    df_gcamp_raw.columns = list(df_gcamp_raw.columns[:4]) + ['RAW_' + col for col in list(df_gcamp_raw.columns[4:])]

    # Add column to store object area
    # get image number associated to well (to retrieve appropriate fragments)
    df_img = pandas.read_csv(os.path.join(segmentsdir, 'gcampsegmentation', 'gcampsegmentation_Image.csv'))
    df_img = df_img[df_img['FileName_gcamp'].str.contains(well)]
    df_img = df_img[['ImageNumber']]
    imgNum = df_img.iloc[0].values[0]

    # get objects and corresponding centers
    df_area = pandas.read_csv(os.path.join(segmentsdir, 'gcampsegmentation', 'gcampsegmentation_neuronFragment.csv'))
    df_area = df_area[df_area['ImageNumber'] == imgNum]
    df_area = df_area[['ObjectNumber', 'AreaShape_Area']]
    df_area = df_area.rename(columns={'ObjectNumber': 'obj_number', 'AreaShape_Area': 'area'})

    df_gcamp_norm = df_gcamp_norm.merge(df_area)
    df_gcamp_raw = df_gcamp_raw.merge(df_area)
    return df_gcamp_norm, df_gcamp_raw


def extractFeaturesFromWell(config, well, controlWellsOnly=False, threshold_multiplier=1.0):
    cxpPrinter.cxpPrint('Computing features for well {0}'.format(well))

    threshold_multiplier = float(threshold_multiplier)

    # get paths and metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    tseriesdir = metadata_dict["config"]["var"]["tseriesdir"]
    outputdir = metadata_dict["config"]["var"]["outputdir"]
    segmentsdir = metadata_dict["config"]["var"]["segmentsdir"]
    resourcesdir = metadata_dict["config"]["var"]["resourcesdir"]
    platename = metadata_dict["config"]["var"]["platename"]
    analysis_type = metadata_dict["config"]["var"]["analysistype"]

    # build dictionary of well types
    well_types_dict = {}
    # well_types_dict['analysis_type'] = analysis_type
    with open(os.path.join(resourcesdir, 'well-mappings', analysis_type + '.csv'), 'r') as f:
        reader = csv.reader(f)
        well_labels = list(reader)
    for well_label in well_labels:
        well_types_dict[well_label[0]] = well_label[1:]

    # get peak threshold
    if controlWellsOnly:
        peak_thresholds_norm = peak_thresholds_raw = [float("-inf"),float("-inf"),float("-inf")]
    else:
        if analysis_type == "standard":
            # use thresholds computed on the fly for given plate
            path_peak_threshold = os.path.join(outputdir, 'peak_threshold.csv')
        else:
            # use pre-computed thresholds
            path_peak_threshold = os.path.join(resourcesdir, 'threshold', 'peak_threshold.csv')

        # read in peak thresholds (normalized and raw)
        with open(path_peak_threshold, 'r') as f:
            reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
            thresholds = list(reader)
        thresholds = numpy.asarray(thresholds) * threshold_multiplier
        peak_thresholds_norm = thresholds[0:3]
        peak_thresholds_raw = thresholds[3:]

    # Read gcamp signals from csv file
    with open(os.path.join(tseriesdir,'{0}_fragments_timeseries.csv'.format(well)), 'r') as f:
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        gcamp_signals = list(reader)

    # if no objects identified in well, no need to process it
    if len(gcamp_signals) < 1:
        return

    # get global extrema of signals (not including background)
    with open(os.path.join(outputdir,'global_extrema.csv'), 'r') as f:
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        extrema = list(reader)
    global_minimum = extrema[0][0]
    global_maximum = extrema[1][0]

    # get well minimum (as plate background minimum value)
    with open(os.path.join(tseriesdir, "{0}_background_timeseries.csv".format(well)), 'r') as f:
        reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
        wellmins = list(reader)[0]

    # normalize signals
    gcamp_signals_normalized = [gcamp_normalize(gcamp, numpy.min(wellmins), global_maximum) for gcamp in gcamp_signals]

    # update output dir if threshold multiplier is provided
    if threshold_multiplier != 1.0:
        outputdir = os.path.join(outputdir, 't'+str(threshold_multiplier))

    # Extract features from well
    number_of_additional_timepoints = 4.0
    gcamp_signals_normalized_interpolated = [gcamp_interpolate(gcamp, number_of_additional_timepoints) for gcamp in gcamp_signals_normalized]
    gcamp_signals_raw_interpolated = [gcamp_interpolate(gcamp, number_of_additional_timepoints) for gcamp in gcamp_signals]
    df_gcamp_norm, df_gcamp_raw = process_well(gcamp_signals_normalized_interpolated, peak_thresholds_norm,
                            gcamp_signals_raw_interpolated, peak_thresholds_raw,
                            len(gcamp_signals[0]), platename, well, well_types_dict,
                            segmentsdir, gcamp_interpolate(wellmins,number_of_additional_timepoints), outputdir)

    # save well features to file
    df_gcamp_norm.to_csv(os.path.join(outputdir, "{0}_features.csv".format(well)), index=False)
    df_gcamp_raw.to_csv(os.path.join(outputdir, "{0}_features_raw.csv".format(well)), index=False)


if __name__ == "__main__":
    config = sys.argv[1]
    well = sys.argv[2]
    threshold_multiplier = sys.argv[3]
    extractFeaturesFromWell(config=config, well=well, threshold_multiplier=threshold_multiplier)
