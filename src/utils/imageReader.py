import os
import numpy
import scipy.misc
import bioformats
import javabridge


def df_apply_make_image_path(df_series):
    """
    :param df_series: a row in the metadata dataframe that contains image path info
    :return: a string that is the path to an image from df_series
    """
    image_path = os.path.abspath(
        os.path.join(df_series["Metadata_pathname"], df_series["Metadata_filename"])
    )

    return image_path


# Returns list of image paths for images associated to a given well
def get_well_image_path_list(metadata_config, df, site = 1):
    # ensure we are looking at the same site across all images
    numSites = df.groupby('Metadata_site')['Metadata_site'].nunique().size
    if numSites > 1:
        raise ValueError('Inconsistent input - More than 1 distinct sites retrieved from images metadata')
    site = numpy.asscalar(df['Metadata_site'][0])

    df_stack = df.loc[(df["Metadata_channel"] == metadata_config["var"]["gcamp_channel_number"]) &
                      (df["Metadata_site"] == site)]
    df_stack.sort_values(["Metadata_timepoint"], ascending=[True])
    image_path_pandas_series = df_stack.apply(df_apply_make_image_path, axis=1)
    image_path_list = image_path_pandas_series.tolist()
    return image_path_list


# Returns stack of images
def read_image_stack(image_path_list):
    # get file extension to read files using appropriate library
    filename, file_extension = os.path.splitext(image_path_list[0])
    if file_extension == '.C01':
        image_list = [read_image_bioformats(image_path) for image_path in image_path_list]
    else:
        image_list = [read_image(image_path) for image_path in image_path_list]

    # stack images
    image_dstack = numpy.dstack(image_list).astype("float")
    return image_dstack


def import_dstack_from_well_dataframe(metadata_config, df, site = 1):
    """
    :param image_path: a string that is the path to an image
    :return: the image found at *image_path*
    """
    image_path_list = get_well_image_path_list(metadata_config, df, site)
    return read_image_stack(image_path_list)


def read_image(image_path):
    img = scipy.misc.imread(image_path)
    if img.ndim == 3:  # assumes RGB format
        img = img[:,:,0]
    return img


def read_image_bioformats(image_path):
    with bioformats.ImageReader(image_path) as reader:
        return reader.read(rescale=False)
