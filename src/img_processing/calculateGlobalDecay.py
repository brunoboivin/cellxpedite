from __future__ import division
import os
import random
import sys
import numpy
import scipy.optimize
from src.utils import imageReader, metadataExtractor, cxpPrinter
import bioformats
import javabridge
from threading import Thread


def calculate_global_decay(config):
    """
    Calculate the global decay that is due to photo-bleaching. A subset of time-series from the plate is used to fit
    an exponential model. The time-series are from randomly selected pixels across the plate. During testing,
    the plating of neurons was low and the majority of pixels
    :param config: The path to the configuration file _\*.cfg_
    :return:
    """
    cxpPrinter.cxpPrint('Calculating global decay')

    # parse config
    metadata_dict = metadataExtractor.import_metadata(config)

    # import the plate metadata
    globdecaydir = metadata_dict["config"]["var"]["globdecaydir"]
    p0 = metadata_dict["config"]["globaldecay"]["p0"]
    max_heap_size = metadata_dict["config"]["javabridge"]["max_heap_size"]
    timelabels = metadata_dict["timelabels"]
    well_names = metadata_dict["well_names"]

    # sample time-series from pixels for a site within each well
    javabridge.start_vm(class_path=bioformats.JARS, max_heap_size=max_heap_size)
    p_tseries_list = [pixel_timeseries_from_well(metadata_dict, well) for well in well_names]
    javabridge.kill_vm()

    # estimate the parameters of decay and export them.
    popt = decay_fit(timelabels, p_tseries_list, p0)
    numpy.savetxt(os.path.join(globdecaydir, "global_exp_model.csv"), popt, delimiter=",")
    return


def decay_fit(timelabels, y_list, p0):
    """
    :param timelabels: The time of image acquisition.
    :param y_list: A list of fluorescence measurements at timelabels.
    :param p0: The initial values to prime the optimization algorithm
    :return popt:
    :return pcov:
    """
    y_array = numpy.vstack(y_list)
    x_array = numpy.ones_like(y_array) * timelabels
    y = y_array.flatten()
    x = x_array.flatten()
    popt, pcov = scipy.optimize.curve_fit(exponential_func, x, y, p0=p0, bounds=(0, numpy.inf))
    return popt


def exponential_func(x, a, b, c):
    """
    :param x: the independent variable. In this case it is the time of image acquisition.
    :param a: The amount of fluorescence at x = 0.
    :param b: The decay constant.
    :param c: The offset. Background fluorescence contributes to this value.
    :return y: The dependent variable.
    """
    y = a*numpy.exp(-b*x)+c
    return y


def pixel_timeseries_from_well(metadata_dict, well):
    """
    :param x: the independent variable. In this case it is the time of image acquisition.
    :param a: The amount of fluorescence at x = 0.
    :param b: The decay constant.
    :param c: The offset. Background fluorescence contributes to this value.
    :return y: The dependent variable.
    """
    df_well_metadata = metadata_dict["df_well_dict"][well]

    # get list of images
    image_path_list = imageReader.get_well_image_path_list(metadata_dict["config"], df_well_metadata)
    totalNumImages = len(image_path_list)

    # get random col/row indices
    firstImg = imageReader.read_image_stack(image_path_list[:1])
    n = metadata_dict["config"]["globaldecay"]["pixelsperwell"]  # num of random pixels per well
    row = random.sample(xrange(firstImg.shape[0]), n)
    col = random.sample(xrange(firstImg.shape[1]), n)

    # define batch size
    batch_size = 50
    startIndex = 0
    endIndex = (startIndex + batch_size) if (startIndex + batch_size) < totalNumImages else totalNumImages

    # initialize time series
    pixelsTimeSeries = numpy.zeros((n, totalNumImages))

    while endIndex <= totalNumImages and startIndex != endIndex:
        # read images and add current max proj to stack
        img_dstack_batch = imageReader.read_image_stack(image_path_list[startIndex:endIndex])

        # process batch time series
        ts_batch = numpy.array(img_dstack_batch[row,col,:])
        for i, pixel_batch_ts in enumerate(ts_batch):
            pixelsTimeSeries[i][startIndex:endIndex] = pixel_batch_ts

        # update indices
        startIndex = endIndex
        endIndex = (startIndex + batch_size) if (startIndex + batch_size) < totalNumImages else totalNumImages

    return pixelsTimeSeries


if __name__ == "__main__":
    calculate_global_decay(sys.argv[1])