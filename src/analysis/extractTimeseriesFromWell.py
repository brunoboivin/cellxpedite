from __future__ import division
from src.utils import imageReader, metadataExtractor, cxpPrinter
import os, sys
import csv
import skimage.io
import numpy
import scipy.misc
import scipy.optimize
import pandas
import bioformats
import javabridge
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random


def curve_fit(func, timelabels, tseries, p0, popt_globaldecay):
    """
    :param func: the parameterized function to be used for fitting
    :param timelabels: the timepoints corresponding to datapoints
    :param tseries: in this case, the measured GCAMP6 fluorescence signal
    :param p0: the estimates for the fitted parameters; starting values for the search algorithm
    :param popt_globaldecay: the global decay rate of fluorescence, previously calculated
    :return: the maximum value of a time-series after it has been corrected
    """
    popt, pcov = scipy.optimize.curve_fit(func, timelabels, tseries, p0=p0)
    tseries_corrected = (tseries - popt[1]) / numpy.exp(-popt_globaldecay[1] * timelabels) + popt[1]
    return tseries_corrected


def model_mono_exp(x, a, b):
    return a * numpy.exp(-b * x)


def fit_model_to_timeseries(x, y, model):
    bounds = (0, numpy.inf)
    opt_parms, parm_cov = scipy.optimize.curve_fit(model, x, y, bounds=bounds)
    return opt_parms, parm_cov


def correct_photobleaching(ts):
    x = numpy.asarray(range(len(ts)))
    opt_parms, parm_cov = fit_model_to_timeseries(x, ts, model_mono_exp)
    fit_curve = model_mono_exp(x, *opt_parms)
    normalized_fit_curve = fit_curve / numpy.max(fit_curve)
    return ts / normalized_fit_curve


def generateFigure(title, output_path, timeseries, bg=None, xlabel='Time', ylabel='Fluorescence intensity'):
    fig = plt.figure(figsize=(11, 8))
    for ts in timeseries:
        plt.plot(ts)
    if bg != None:
        plt.plot(bg, c='black', linewidth=3.0)
    plt.title(title, loc='left', fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(prop={'size': 14})
    fig.savefig(output_path)
    plt.close(fig)


def extractTimeseriesFromWell(config, well, segmentsfilepattern="{0}_maxprojection_neuronFragments.tiff"):
    cxpPrinter.cxpPrint('Extracting time series from well {0}'.format(well))

    # parse config to get plate metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    segmentsdir = metadata_dict["config"]["var"]["segmentsdir"]
    illumpath = metadata_dict["config"]["var"]["illumpath"]
    globdecaydir = metadata_dict["config"]["var"]["globdecaydir"]
    tseriesdir = metadata_dict["config"]["var"]["tseriesdir"]
    figuresdir = metadata_dict["config"]["var"]["figuresdir"]
    max_heap_size = metadata_dict["config"]["javabridge"]["max_heap_size"]
    timelabels = metadata_dict["timelabels"]

    # read illum corr file
    img_illum = scipy.misc.imread(illumpath)

    # read segments from image and build map (ObjectNumber,Points)
    segments_img = scipy.misc.imread(os.path.join(segmentsdir, "gcampsegmentation", segmentsfilepattern.format(well)))
    d = {}
    for i in numpy.unique(segments_img[segments_img > 0]):
        d[i] = numpy.nonzero(segments_img == i)

    # define function for curve_fit as an inner function
    popt_globaldecay = pandas.read_csv(os.path.join(globdecaydir,"global_exp_model.csv"),sep=',',header=None)
    popt_globaldecay = popt_globaldecay.values
    p0 = (popt_globaldecay[0], popt_globaldecay[2])


    def exp_offset_func(x, a, c):
        """
        :param x: the timepoint
        :param a: the initial value
        :param c: the offset
        :return: in this case, the GCAMP6 fluorescence signal
        """
        return a * numpy.exp(-popt_globaldecay[1] * x) + c


    # get well metadata and start java vm
    df_well_metadata = metadata_dict["df_well_dict"][well]
    javabridge.start_vm(class_path=bioformats.JARS, max_heap_size=max_heap_size)

    # get list of images
    image_path_list = imageReader.get_well_image_path_list(metadata_dict["config"], df_well_metadata)
    totalNumImages = len(image_path_list)

    # read well background time series if available; otherwise random select coords of background pixels
    background_ts_available = (segmentsfilepattern != "{0}_maxprojection_neuronFragments.tiff")
    if background_ts_available:
        with open(os.path.join(tseriesdir, "{0}_background_timeseries.csv".format(well)), 'r') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            background_intensity = numpy.asarray(list(reader)[0])
    else:
        # randomly select background pixels
        numPixels = 10000  # number of random pixels
        background_coords = numpy.nonzero(segments_img == 0)

        # ensure sample size is <= population size
        if len(background_coords[0]) < numPixels:
            numPixels = len(background_coords[0])

        # randomly select numPixels of indices
        randIndices = random.sample(range(len(background_coords[0])), numPixels)
        background_coords = (background_coords[0][randIndices],background_coords[1][randIndices])

        # initialize background time series
        backgroundTimeSeries = numpy.zeros((numPixels,totalNumImages))


    # initialize pixel time series
    pixelsTimeSeries = []
    for k in sorted(d.keys()):
        objectKpixels = numpy.zeros((len(d[k][0]),totalNumImages))
        pixelsTimeSeries.append(objectKpixels)
    pixelsTimeSeries = numpy.asarray(pixelsTimeSeries)

    # define batch size
    batch_size = 50
    startIndex = 0
    endIndex = (startIndex + batch_size) if (startIndex + batch_size) < totalNumImages else totalNumImages

    # process images in batches
    while endIndex <= totalNumImages and startIndex != endIndex:
        # read images and add current max proj to stack
        img_dstack_batch = imageReader.read_image_stack(image_path_list[startIndex:endIndex])

        # correct for illumination
        img_dstack_batch_corrected = img_dstack_batch / img_illum[:, :, None]

        # get objects batch time series
        ts_batch = [img_dstack_batch_corrected[d[k]] for k in sorted(d.keys())]
        for o,obj in enumerate(ts_batch):
            for p,pixel_batch_ts in enumerate(obj):
                pixelsTimeSeries[o][p][startIndex:endIndex] = pixel_batch_ts

        # background batch time series
        if not background_ts_available:
            ts_background_batch = img_dstack_batch_corrected[background_coords]
            for i, bg_pixel_batch_ts in enumerate(ts_background_batch):
                backgroundTimeSeries[i][startIndex:endIndex] = bg_pixel_batch_ts

        # update indices
        startIndex = endIndex
        endIndex = (startIndex + batch_size) if (startIndex + batch_size) < totalNumImages else totalNumImages

    # close java vm
    javabridge.kill_vm()

    # time series prior to photobleaching/bckg correction
    fragmentsTimeSeries = [
        numpy.mean(
            [ts for ts in pixelsTimeSeries[i]],
            axis=0
        )
        for i in range(len(d.keys()))
    ]

    # save figure pre-correction
    generateFigure(
        title='Well: {0}, Original signals'.format(well),
        output_path=os.path.join(figuresdir,'{0}_original.png'.format(well)),
        timeseries=fragmentsTimeSeries
    )
    # save unprocessed time series
    with open(os.path.join(tseriesdir, "{0}_fragments_timeseries_unprocessed.csv".format(well)), "w") as f:
        writer = csv.writer(f)
        writer.writerows(fragmentsTimeSeries)

    # mean background intensity
    if not background_ts_available:
        background_intensity = numpy.mean(
            [curve_fit(exp_offset_func, timelabels, ts, p0, popt_globaldecay) for ts in backgroundTimeSeries],
            axis=0
        )

    # save figure post-correction
    generateFigure(
        title='Well: {0}, Signals after photobleaching correction'.format(well),
        output_path=os.path.join(figuresdir,'{0}_after_photo_corr.png'.format(well)),
        timeseries=fragmentsTimeSeries
    )

    # save figure final signals
    generateFigure(
        title='Well: {0}, Signals after all corrections'.format(well),
        output_path=os.path.join(figuresdir,'{0}_final.png'.format(well)),
        timeseries=fragmentsTimeSeries
    )

    # Save time series to file
    with open(os.path.join(tseriesdir, "{0}_background_timeseries.csv".format(well)), "w") as f:
        writer = csv.writer(f)
        writer.writerows([background_intensity])

    with open(os.path.join(tseriesdir, "{0}_fragments_timeseries.csv".format(well)), "w") as f:
        writer = csv.writer(f)
        writer.writerows(fragmentsTimeSeries)


if __name__ == "__main__":
    extractTimeseriesFromWell(*sys.argv[1:])
