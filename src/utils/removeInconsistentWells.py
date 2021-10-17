from __future__ import division
import sys, os
import configparser
import os, glob
from src.utils import cxpPrinter, metadataExtractor


def removeWellsData(dir, wells, numPoints):
    inconsistent_wells = []
    channel = 'd2'
    for well in wells:
        if len(well) > 0:
            well_files = glob.glob(dir + '/*' + well + '*' + channel + '.*')
            if len(well_files) < numPoints:
                inconsistent_wells.append(well)
            elif len(well_files) > numPoints:
                inconsistent_wells.append(well)
    return inconsistent_wells


# To remove wells that have more or less than 45 time points
def removeInconsistentWells(config):
    cxpPrinter.cxpPrint('Checking for inconsistent wells')

    # expected number of time points for standard recordings
    numPoints = 45

    # parse config file
    cfgparser = configparser.ConfigParser()
    cfgparser.read(config)
    inputdir = cfgparser.get("var", "imagedir")
    metadatadir = cfgparser.get("var", "metadatadir")

    # read in list of well
    metadata_dict = metadataExtractor.import_metadata(config)
    well_names = metadata_dict["well_names"]

    # remove inconsistent wells
    inconsistent_wells = removeWellsData(inputdir, well_names, numPoints)

    # save list of incomplete wells
    wells_removed_log_file = open(os.path.join(metadatadir, 'wells_inconsistent.txt'), 'w')
    for w in inconsistent_wells:
        wells_removed_log_file.write("%s\n" % w)


if __name__ == "__main__":
    removeInconsistentWells(*sys.argv[1:])
