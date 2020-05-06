from __future__ import division
import numpy as np
import pandas as pd
import sys, os, csv
from src.utils import metadataExtractor, cxpPrinter
from src.analysis import extractFeaturesFromWell
from skimage.filters import threshold_otsu


def getPeakThreshold(config,wellmapping):
    cxpPrinter.cxpPrint('Calculating peak threshold from control wells')

    # get paths and metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    outputdir = metadata_dict["config"]["var"]["outputdir"]
    resourcesdir = metadata_dict["config"]["var"]["resourcesdir"]

    # get list of control wells
    with open(os.path.join(resourcesdir,'well-mappings', wellmapping + '.csv'), 'r') as f:
        reader = csv.reader(f)
        control_wells = list(reader)
    control_wells = control_wells[0][1:] + control_wells[1][1:]

    # ensure well data is available; compromised data might have been removed
    actualWells = metadata_dict["well_names"]
    control_wells = [well for well in control_wells if well in actualWells]

    # perform (minimal) feature extraction on control wells3
    for well in control_wells:
        extractFeaturesFromWell.extractFeaturesFromWell(config, well, controlWellsOnly=True)

    # merge data from control wells
    dataframes_norm = [pd.read_csv(os.path.join(outputdir,"{0}_features.csv".format(well))) for well in control_wells]
    df_plate_norm = pd.concat(dataframes_norm)
    dataframes_raw = [pd.read_csv(os.path.join(outputdir,"{0}_features_raw.csv".format(well))) for well in control_wells]
    df_plate_raw = pd.concat(dataframes_raw)

    # compute thresholds from control wells
    threshold_labels_norm = ["WM_amplitude","SM_amplitude","LM_amplitude"]
    threshold_labels_raw = ["RAW_WM_amplitude", "RAW_SM_amplitude", "RAW_LM_amplitude"]

    thresholds = []
    for label in threshold_labels_norm:
        if label in df_plate_norm.columns:
            thresholds.append(threshold_otsu(df_plate_norm[label].dropna(axis=0)))
        else:
            thresholds.append(0)
            cxpPrinter.cxpPrint("Threshold label '" + label + "' not found; assigned default threshold 0.")

    for label in threshold_labels_raw:
        if label in df_plate_raw.columns:
            thresholds.append(threshold_otsu(df_plate_raw[label].dropna(axis=0)))
        else:
            thresholds.append(0)
            cxpPrinter.cxpPrint("Threshold label '" + label + "' not found; assigned default threshold 0.")

    # save threshold
    np.savetxt(os.path.join(outputdir, "peak_threshold.csv"), thresholds)

    # remove control well csv files
    for well in control_wells:
        os.remove(os.path.join(outputdir,"{0}_features.csv".format(well)))
        os.remove(os.path.join(outputdir, "{0}_features_raw.csv".format(well)))


if __name__ == "__main__":
    getPeakThreshold(*sys.argv[1:])
