from __future__ import division
import pandas as pd
import sys, os
from src.utils import metadataExtractor

"""
Used to reduce fragment features to minimal information needed for
merging the fragments, i.e. (ImageNumber, ObjectNumber, Location_Center_X, Location_Center_Y).
** This significantly reduces computing time.
"""
def downsizeFragmentFeatures(config):
    # parse config to get plate metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    segmentsdir = metadata_dict["config"]["var"]["segmentsdir"]

    df = pd.read_csv(os.path.join(segmentsdir, "gcampsegmentation", 'gcampsegmentation_neuronFragment.csv'))
    df = df[['ImageNumber','ObjectNumber', 'Location_Center_X', 'Location_Center_Y']]
    df.to_csv(os.path.join(segmentsdir, "gcampsegmentation", 'gcampsegmentation_neuronFragment_downsized.csv'), index=False)


if __name__ == "__main__":
    downsizeFragmentFeatures(sys.argv[1])