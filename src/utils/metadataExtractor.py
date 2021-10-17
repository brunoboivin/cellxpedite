import configparser
import json
import os
import re
import sys
import numpy
import pandas
from src.utils import cxpPrinter


def createMetadataFiles(config):
    """
    Find the images to be processed and create a CSV that contains all the metadata.
    Then create an additional CSV for each well represented in the images.
    Data is processed one plate at a time.
    :param config: The path to the configuration file _\*.cfg_
    :return:
    """
    cxpPrinter.cxpPrint('Creating metadata files')

    # parse config
    cfgparser = configparser.ConfigParser()
    cfgparser.read(config)
    imagedir = cfgparser.get("var", "imagedir")
    metadatadir = cfgparser.get("var", "metadatadir")
    image_names_pattern = cfgparser.get("re", "image_names_pattern")
    wellnamespath = cfgparser.get("var", "wellnamespath")

    # identify images and corresponding metadata
    image_names = identify_image_set(imagedir, image_names_pattern)
    df_image_metadata_cp = identify_image_metadata(imagedir, image_names, image_names_pattern)
    # export CSVs
    export_loaddata_csv_plate(df_image_metadata_cp, metadatadir)
    export_loaddata_csv_well(df_image_metadata_cp, metadatadir)

    # create a text file listing all the wells
    write_well_textfile(df_image_metadata_cp, wellnamespath)

    return


def write_well_textfile(df_image_metadata_cp, wellnamespath):
    well_names = pandas.unique(df_image_metadata_cp["Metadata_well"])
    with open(wellnamespath, "w") as f:
        for well in well_names:
            f.write(well + os.linesep)


def export_loaddata_csv_plate(df_image_metadata_cp, outputdir):
    """
    The dataframe is saved as a CSV into the *outputdir*.
    :param df_image_metadata_cp:
    :param outputdir:
    :return:
    """
    df_image_metadata_cp.to_csv(os.path.join(outputdir, "metadata_plate.csv"), index=False)
    return


def export_loaddata_csv_well(df_image_metadata_cp, outputdir):
    """
    The dataframe is saved as a collection of CSVs, sorted by well, into the *outputdir*.
    :param df_image_metadata_cp:
    :param outputdir:
    :return:
    """
    well_names = pandas.unique(df_image_metadata_cp["Metadata_well"])
    for well in well_names:
        is_well = (df_image_metadata_cp["Metadata_well"] == well)
        df_well = df_image_metadata_cp.loc[is_well]
        df_well.to_csv(os.path.join(outputdir, "metadata_{0}.csv".format(well)), index=False)
    return


def identify_image_metadata(imagedir, image_names, image_names_pattern):
    """
    Extract metadata from the filenames in *image_names* list.
    :param imagedir:
    :param image_names:
    :param image_names_pattern:
    :return: a pandas dataframe with image filenames and corresponding metadata
    """
    # find metadata from the image filenames
    image_metadata = [re_identify_image_metadata(fn, image_names_pattern) for fn in image_names]
    image_metadata = [list(metadata) for metadata in image_metadata if metadata is not None]
    image_metadata = [[fn] + metadata for fn, metadata in zip(image_names, image_metadata)]
    # convert metadata from a list of lists into a pandas dataframe
    # The order of the metadata in _image_metadata_ matches the order of groups in the regular expression
    # The column names must match the order of the metadata lists within _image_metadata_
    match = re.match(image_names_pattern, image_names[0])
    gd = match.re.groupindex
    image_metadata_columns = [None] * len(gd)  # initialize list so it can treated as an array
    for key in gd:
        image_metadata_columns[gd[key]-1] = key
    image_metadata_columns.insert(0, "filename")
    df_image_metadata = pandas.DataFrame(data=image_metadata, columns=image_metadata_columns)
    df_image_metadata[["timepoint", "site", "channel"]] = \
        df_image_metadata[["timepoint", "site", "channel"]].astype(numpy.int64)
    # Change the names of the columns to be CellProfiler compatible
    df_image_metadata_cp = df_image_metadata.rename(
        columns={"filename": "Metadata_filename",
                 "plate": "Metadata_plate",
                 "timepoint": "Metadata_timepoint",
                 "well": "Metadata_well",
                 "site": "Metadata_site",
                 "channel": "Metadata_channel"})
    df_image_metadata_cp["Metadata_pathname"] = imagedir
    return df_image_metadata_cp


def identify_image_set(imagedir, image_names_pattern):
    """
    Find all the images within the *imagedir*.
    :param imagedir:
    :param image_names_pattern:
    :return: a list of image names that are part of the image set
    """
    image_names_from_os = sorted(os.listdir(imagedir))
    image_names = [re_identify_image_set(fn, image_names_pattern) for fn in image_names_from_os]
    image_names = [name for name in image_names if name is not None]
    return image_names


def re_identify_image_metadata(filename, image_names_pattern):
    """
    Apply a regular expression to the *filename* and return metadata
    :param filename:
    :param image_names_pattern:
    :return: a list with metadata derived from the image filename
    """
    match = re.match(image_names_pattern, filename)
    return None if match is None else match.groups()


def re_identify_image_set(filename, image_names_pattern):
    """
    Apply a regular expression to the *filename* and return the filename if it matches.
    :param filename:
    :param image_names_pattern:
    :return: the filename that matches the input pattern
    """
    match = re.match(image_names_pattern, filename)
    return None if match is None else filename


def import_metadata(config):
    # add the config file to the dictionary
    cfgparser = configparser.ConfigParser()

    cfgparser.read(config)

    metadata_dict = {
        "config": {s: dict(cfgparser.items(s)) for s in cfgparser.sections()}
    }

    metadata_dict["config"]["globaldecay"]["pixelsperwell"] = int(metadata_dict["config"]["globaldecay"]["pixelsperwell"])
    metadata_dict["config"]["globaldecay"]["p0"] = json.loads(metadata_dict["config"]["globaldecay"]["p0"])
    metadata_dict["config"]["var"]["gcamp_channel_number"] = int(metadata_dict["config"]["var"]["gcamp_channel_number"])

    # add additional metadata to the dictionary
    metadatadir = os.path.abspath(metadata_dict["config"]["var"]["metadatadir"])
    metadata_dict["config"]["var"]["metadatadir"] = metadatadir
    df_image_metadata = pandas.read_csv(os.path.join(metadatadir, "metadata_plate.csv"))

    timelabels = pandas.unique(df_image_metadata["Metadata_timepoint"])
    timelabels = numpy.sort(timelabels) - 1

    well_names = pandas.unique(df_image_metadata["Metadata_well"])

    metadata_dict["df_image_metadata"] = df_image_metadata
    metadata_dict["df_well_dict"] = {well: pandas.read_csv(os.path.join(metadatadir, "metadata_{0}.csv".format(
        well))) for well in well_names}
    metadata_dict["timelabels"] = timelabels
    metadata_dict["well_names"] = well_names

    return metadata_dict


# To remove specified wells from further analysis
def removeWellsFromWellList(wells_to_remove, config):
    # parse config
    cfgparser = configparser.ConfigParser()
    cfgparser.read(config)
    wellnamespath = cfgparser.get("var", "wellnamespath")
    metadatadir = cfgparser.get("var", "metadatadir")

    # Step 1: Update plate metadata file
    # read plate metadata
    df_image_metadata = pandas.read_csv(os.path.join(metadatadir, "metadata_plate.csv"))

    # remove rows where well matches any of wells_to_remove
    df_image_metadata = df_image_metadata[~df_image_metadata['Metadata_well'].isin(wells_to_remove)]

    # save file
    df_image_metadata.to_csv(os.path.join(metadatadir, "metadata_plate.csv"), index=False)

    # Step 2: Update list of wells in 'well_names.txt'
    write_well_textfile(df_image_metadata, wellnamespath)

    # save list of wells removed
    wells_removed_log_file = open(os.path.join(metadatadir,'wells_removed.txt'), 'w')
    for w in wells_to_remove:
        wells_removed_log_file.write("%s\n" % w)

    # print list of wells removed
    if len(wells_to_remove) > 0:
        cxpPrinter.cxpPrint('The following wells were removed: ' + str(wells_to_remove))
    else:
        cxpPrinter.cxpPrint('No wells were removed')


if __name__ == "__main__":
    createMetadataFiles(sys.argv[1])
