from __future__ import division
import sys, os
import configparser
import csv, os, glob
from src.utils import cxpPrinter


def removeWellsData(dir,wells):
    for well in wells:
        if len(well) > 0:
            well_files = glob.glob(dir + '/*' + well + '*')
            for f in well_files:
                os.remove(os.path.join(dir, f))


# To remove wells known to be compromised
# - Mapping from platename (folder name) to list of wells must be provided
#   in the res/compromised_wells.csv file
def removeCompromisedWells(config):
    cxpPrinter.cxpPrint('Checking for compromised wells')

    # parse config file
    cfgparser = configparser.ConfigParser()
    cfgparser.read(config)
    platename = cfgparser.get("var", "platename")
    inputdir = cfgparser.get("var", "imagedir")
    resourcesdir = cfgparser.get("var", "resourcesdir")
    metadatadir = cfgparser.get("var", "metadatadir")

    # mapping from plate names to corresponding list of compromised wells
    compromised_dict = {}
    with open(os.path.join(resourcesdir, 'compromised_wells.csv'), 'r') as f:
        reader = csv.reader(f)
        compromised_list = list(reader)
    for compromised_item in compromised_list:
        compromised_dict[compromised_item[0]] = compromised_item[1:]

    # remove compromised data from input folder
    if platename in compromised_dict.keys():
        compromised_wells = compromised_dict[platename]
        if len(compromised_wells) > 0:
            removeWellsData(inputdir, compromised_wells)

            # save list of compromised wells
            wells_removed_log_file = open(os.path.join(metadatadir, 'wells_compromised.txt'), 'w')
            for w in compromised_wells:
                wells_removed_log_file.write("%s\n" % w)

            cxpPrinter.cxpPrint('The following wells were deemed compromised and removed: ' + str(compromised_wells))
        else:
            cxpPrinter.cxpPrint('No compromised data to remove')
    else:
        cxpPrinter.cxpPrint('No compromised data to remove')


if __name__ == "__main__":
    removeCompromisedWells(*sys.argv[1:])