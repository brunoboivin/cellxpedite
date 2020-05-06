from __future__ import division
import pandas as pd
import sys, os, csv
from src.utils import metadataExtractor, cxpPrinter

def generateZfactorsFile(input_filepath,output_filpath,wells,platename):
    neg_control_label = wells[0][0]
    neg_control_wells = wells[0][1:]
    pos_control_label = wells[1][0]
    pos_control_wells = wells[1][1:]

    # read control well features
    df_plate = pd.read_csv(input_filepath)
    df_neg_ctrl = df_plate[(df_plate["well"].isin(neg_control_wells))]
    df_pos_ctrl = df_plate[(df_plate["well"].isin(pos_control_wells))]

    # compute z-factor for all features and save to file
    df_zfactor = pd.Series()
    df_zfactor['plate'] = platename
    df_zfactor[neg_control_label] = '-'.join(neg_control_wells)
    df_zfactor[pos_control_label] = '-'.join(pos_control_wells)
    df_zfactor = df_zfactor.append(1 - (3*(df_pos_ctrl.std(ddof=0) + df_neg_ctrl.std(ddof=0)) / abs(df_pos_ctrl.mean() - df_neg_ctrl.mean())))
    df_zfactor.to_csv(output_filpath)


def computeZfactor(config, threshold_multiplier=1.0):
    cxpPrinter.cxpPrint('Computing Z factors')

    # get paths and metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    outputdir = metadata_dict["config"]["var"]["outputdir"]
    resourcesdir = metadata_dict["config"]["var"]["resourcesdir"]
    platename = metadata_dict["config"]["var"]["platename"]
    analysis_type = metadata_dict["config"]["var"]["analysistype"]

    # update output dir if threshold multiplier is provided
    threshold_multiplier = float(threshold_multiplier)
    if threshold_multiplier != 1.0:
        outputdir = os.path.join(outputdir, 't'+str(threshold_multiplier))

    # get labels and wells --> assumes neg_ctrl wells on 1st line and pos_ctrl on 2nd line
    with open(os.path.join(resourcesdir, 'well-mappings', analysis_type + '.csv'), 'r') as f:
        reader = csv.reader(f)
        wells = list(reader)

    # normalized data
    generateZfactorsFile(
        input_filepath=os.path.join(outputdir, "{0}_plate_features_wells.csv".format(platename)),
        output_filpath=os.path.join(outputdir,"{0}_zfactors.csv".format(platename)),
        wells=wells,
        platename=platename
    )

    # raw data
    generateZfactorsFile(
        input_filepath=os.path.join(outputdir, "{0}_plate_features_wells_raw.csv".format(platename)),
        output_filpath=os.path.join(outputdir,"{0}_zfactors_raw.csv".format(platename)),
        wells=wells,
        platename=platename
    )


if __name__ == "__main__":
    computeZfactor(*sys.argv[1:])
