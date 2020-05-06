from __future__ import division
import os, sys
import numpy as np
import pandas as pd
from src.utils import imageReader, metadataExtractor, cxpPrinter


# To remove wells where object identification is deemed a failure
# - uses a hardcoded maximum threshold on number of objects
def removeSubstandardWells(config, force_remove_wells=[]):
    cxpPrinter.cxpPrint('Removing substandard wells')

    if len(force_remove_wells) > 1:
        force_remove_wells = force_remove_wells.split(',')
    elif len(force_remove_wells) == 1:
        force_remove_wells = [force_remove_wells]

    # parse config
    metadata_dict = metadataExtractor.import_metadata(config)

    # import the plate metadata
    segmentsdir = metadata_dict["config"]["var"]["segmentsdir"]
    well_names = metadata_dict["well_names"]

    # read in summary of segmentation
    segment_summary_file = 'gcampsegmentation_Image.csv'
    df = pd.read_csv(os.path.join(segmentsdir, 'gcampsegmentation', segment_summary_file))

    # retrieve substandard rows and parse out corresponding well names
    minNumNeurons = 5
    maxNumFragments = 1000
    df = df[(df['Count_neuronFragment'] < minNumNeurons) | (df['Count_neuronFragment'] > maxNumFragments)]
    filenames = list(df['FileName_gcamp'])
    wells_to_remove = [f.split('_')[0] for f in filenames] + force_remove_wells

    # update metadata files to prevent further analysis on these substandard wells
    metadataExtractor.removeWellsFromWellList(wells_to_remove, config)


if __name__ == "__main__":
    removeSubstandardWells(sys.argv[1], *sys.argv[2:])
