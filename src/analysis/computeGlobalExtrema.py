from __future__ import division
import sys, os, csv
from src.utils import metadataExtractor, cxpPrinter
import numpy as np


def computeGlobalExtrema(config):
    cxpPrinter.cxpPrint('Computing plate global extrema')

    # get paths and metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    outputdir = metadata_dict["config"]["var"]["outputdir"]
    tseriesdir = metadata_dict["config"]["var"]["tseriesdir"]
    well_names = metadata_dict["well_names"]

    # Read gcamp signals from csv files
    well_minimums = np.array([])
    well_maximums = np.array([])
    for well in well_names:
        with open(os.path.join(tseriesdir,'{0}_fragments_timeseries.csv'.format(well)), 'r') as f:
            reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
            ts = np.asarray(list(reader))
            if ts.size > 1:
                ts = ts.flatten()
                well_minimums = np.append(well_minimums, min(ts))
                well_maximums = np.append(well_maximums, max(ts))

    # save global extrema
    global_min = min(well_minimums)
    global_max = max(well_maximums)
    np.savetxt(os.path.join(outputdir, "global_extrema.csv"), [global_min,global_max])


if __name__ == "__main__":
    computeGlobalExtrema(sys.argv[1])