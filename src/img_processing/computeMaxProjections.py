import os
import click
import numpy
import skimage.io
from src.utils import imageReader, metadataExtractor, cxpPrinter
import bioformats
import javabridge
import scipy.misc


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.argument("well")
def command(config, well):
    """A program that creates a maximum projection of images found in a WELL from the plate defined by the CONFIG
    file."""
    process_well_maxproj(config, well)


def process_well_maxproj(config, well):
    """
    Calculate the global decay that is due to photo-bleaching. A subset of time-series from the plate is used to fit
    an exponential model. The time-series are from randomly selected pixels across the plate.
    :param config: The path to the configuration file _\*.cfg_
    :param well: The name of a well that is a string
    :return:
    """
    cxpPrinter.cxpPrint("Calculating max projections for well " + well)

    # parse config file and import metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    maxprojdir = metadata_dict["config"]["var"]["maxprojdir"]
    max_heap_size = metadata_dict["config"]["javabridge"]["max_heap_size"]

    # read illum correction file
    illumpath = metadata_dict["config"]["var"]["illumpath"]
    img_illum = scipy.misc.imread(illumpath)  # do not use skimage.io.imread here, dims don't work

    # identify images/metadata for a given well
    df_well_metadata = metadata_dict["df_well_dict"][well]
    javabridge.start_vm(class_path=bioformats.JARS, max_heap_size=max_heap_size)

    # get list of images
    image_path_list = imageReader.get_well_image_path_list(metadata_dict["config"], df_well_metadata)
    totalNumImages = len(image_path_list)

    # define batch size
    batch_size = 50
    startIndex = 0
    endIndex = (startIndex + batch_size) if (startIndex + batch_size) < totalNumImages else totalNumImages

    # initial maxImg to zeros based on size of 1st img
    firstImg = imageReader.read_image_stack(image_path_list[:1])
    image_max_projection = numpy.zeros(firstImg.shape[:2])

    while endIndex <= totalNumImages and startIndex != endIndex:
        # read images and add current max proj to stack
        img_dstack_batch = imageReader.read_image_stack(image_path_list[startIndex:endIndex])

        # correct for illumination
        img_dstack_batch_corrected = img_dstack_batch / img_illum[:,:,None]

        # add current max proj to stack
        img_dstack_batch_corrected = numpy.dstack((image_max_projection, img_dstack_batch_corrected))

        # process images
        image_max_projection = computeMaxProjections(img_dstack_batch_corrected)

        # update indices
        startIndex = endIndex
        endIndex = (startIndex + batch_size) if (startIndex + batch_size) < totalNumImages else totalNumImages


    # close java vm
    javabridge.kill_vm()

    new_filename = os.path.join(maxprojdir, "{0}_maxprojection.tif".format(well))
    skimage.io.imsave(new_filename, image_max_projection)
    return


def computeMaxProjections(img_dstack):
    return img_dstack.max(axis=2).astype('uint16')


if __name__ == "__main__":
    command()