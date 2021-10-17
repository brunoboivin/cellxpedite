from __future__ import division
import os
import sys
import re
import configparser
from src.utils import cxpPrinter


# Get list of files per well for specified input directory
def update_well_files_mapping(input_dir, well_files, filename_pattern):
    accepted_extensions = ['tif', 'tiff', 'c01']
    # get list of img filenames
    files = [f for f in os.listdir(input_dir)
             if os.path.isfile(os.path.join(input_dir, f))
             and any(ext in f.lower() for ext in accepted_extensions)]

    # extract well names
    well_names = []
    pattern = re.compile(filename_pattern)
    for f in files:
        match = pattern.match(f)
        if match is not None:
            well_name = match.group("well")
            if well_name not in well_names:
                well_names.append(well_name)

    # construct full path to files
    files = [os.path.join(input_dir, f) for f in files]

    # update well files mapping
    for well in well_names:
        well_files[well] = [f for f in files if well in f.split('.')[0]]

    return well_files


# To merge data from same plate spread over multiple folders
def mergeFolders(folder1, folder2, output_dir, config):
    cxpPrinter.cxpPrint('Merging data from input folders')

    # parse config file
    cfgparser = configparser.ConfigParser()
    cfgparser.read(config)
    filename_pattern = cfgparser.get("re", "image_names_pattern")

    # create mapping from wells to corresponding most recent set of files
    well_files = {}
    well_files = update_well_files_mapping(folder1, well_files, filename_pattern)
    well_files = update_well_files_mapping(folder2, well_files, filename_pattern)

    # flatten list of files contained in mapping
    flatten = lambda l: [item for sublist in l for item in sublist]
    files = flatten(well_files.values())

    # create output dir
    suffix = '_combined'
    output_folder = os.path.join(output_dir, os.path.basename(os.path.normpath(folder2)) + suffix)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # move and rename files (only channel d2 data for now)
    def renameFile(f, folder1, folder2, output_folder):
        folder1_name = os.path.basename(os.path.normpath(folder1))
        folder2_name = os.path.basename(os.path.normpath(folder2))
        new_filename = os.path.basename(f).replace(folder1_name, folder2_name + suffix)
        return os.path.join(output_folder, new_filename)
    [os.rename(f, renameFile(f, folder1, folder2, output_folder)) for f in files if 'd2.' in f.lower()]


if __name__ == "__main__":
    mergeFolders(*sys.argv[1:])
