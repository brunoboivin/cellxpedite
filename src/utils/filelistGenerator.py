from __future__ import division
import os, sys
import math
from src.utils import metadataExtractor, cxpPrinter


def illumFilelist(input_dir, output_dir, config, r=1.0):
    cxpPrinter.cxpPrint('Generating illumination file list')
    r = float(r)

    # helper functions
    get_sublist_indices = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]  # Bresenham's line algorithm
    flatten = lambda l: [item for sublist in l for item in sublist]

    # get list of image files in input dir
    accepted_extensions = ['tif','tiff','c01']
    files = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files = [f for f in files if any(ext in f.lower() for ext in accepted_extensions)]

    # get list of files per well
    metadata_dict = metadataExtractor.import_metadata(config)
    well_names = metadata_dict["well_names"]
    well_files = {well: sorted([f for f in files if well in f.split('.')[0]]) for well in well_names}

    # apply ratio r to retain appropriate amount of evenly-spaced images over recording
    for well in well_names:
        numFiles = len(well_files[well])
        numFilesToRetain = int(math.floor(r * numFiles))
        indicesToRetain = get_sublist_indices(numFilesToRetain, numFiles)
        well_files[well] = [well_files[well][i] for i in indicesToRetain]

    # merge lists together, sort, and write to file
    files = flatten(sorted(well_files.values()))
    with open(os.path.join(output_dir, 'illum_filelist.txt'), 'w') as out_file:
        for f in files:
            out_file.write("%s\n" % f)


if __name__ == "__main__":
    illumFilelist(*sys.argv[1:])
