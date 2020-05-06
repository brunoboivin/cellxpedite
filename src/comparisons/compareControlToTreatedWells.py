from __future__ import division
import pandas as pd
import numpy as np
import os, sys, csv
from src.utils import metadataExtractor, cxpPrinter


def getPandaSeriesMean(df,platename,wells,well_type):
    ps = pd.Series()
    ps['plate'] = platename
    ps['wells'] = '-'.join(wells)
    ps['well_type'] = well_type
    ps = ps.append(df.mean())
    return ps


def getPandaSeriesSEM(df,platename,wells,well_type):
    ps = pd.Series()
    ps['plate'] = platename
    ps['wells'] = '-'.join(wells)
    ps['well_type'] = well_type
    ps = ps.append(df.sem())
    return ps


def generateComparisonFile(input_filepath, output_filepath, wells, platename):
    control_label = wells[0][0]
    control_wells = wells[0][1:]
    treated_label = wells[1][0]
    treated_wells = wells[1][1:]

    # read control well features
    df_plate = pd.read_csv(input_filepath)

    # control wells
    df_control = df_plate[(df_plate["well"].isin(control_wells))]
    control_wells_used = list(df_control["well"])
    ps_control_mean = getPandaSeriesMean(df_control, platename, control_wells_used,control_label)
    ps_control_sem = getPandaSeriesSEM(df_control, platename, control_wells_used,control_label)

    # treated wells
    df_treated = df_plate[(df_plate["well"].isin(treated_wells))]
    treated_wells_used = list(df_treated["well"])
    ps_treated_mean = getPandaSeriesMean(df_treated, platename, treated_wells_used,treated_label)
    ps_treated_sem = getPandaSeriesSEM(df_treated, platename, treated_wells_used,treated_label)

    df_final = pd.concat([ps_control_mean, ps_control_sem, ps_treated_mean, ps_treated_sem], axis=1)
    df_final.columns = [control_label + ' [MEAN]', control_label + ' [SEM]',
                        treated_label + ' [MEAN]', treated_label + ' [SEM]']
    df_final.to_csv(output_filepath)


def compareControlToTreatedWells(config, threshold_multiplier=1.0):
    cxpPrinter.cxpPrint('Comparing control and treated wells')

    # get paths and metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    outputdir = metadata_dict["config"]["var"]["outputdir"]
    resourcesdir = metadata_dict["config"]["var"]["resourcesdir"]
    platename = metadata_dict["config"]["var"]["platename"]
    analysistype = metadata_dict["config"]["var"]["analysistype"]

    # update output dir if threshold multiplier is provided
    threshold_multiplier = float(threshold_multiplier)
    if threshold_multiplier != 1.0:
        outputdir = os.path.join(outputdir, 't' + str(threshold_multiplier))

    # get list of control wells
    with open(os.path.join(resourcesdir, 'well-mappings', analysistype + '.csv'), 'r') as f:
        reader = csv.reader(f)
        wells = list(reader)

    # normalized data
    generateComparisonFile(
        os.path.join(outputdir, "{0}_plate_features_wells.csv".format(platename)),
        os.path.join(outputdir, "{0}_features_control_vs_treated_wells.csv".format(platename)),
        wells,
        platename
    )

    # raw data
    generateComparisonFile(
        os.path.join(outputdir, "{0}_plate_features_wells_raw.csv".format(platename)),
        os.path.join(outputdir, "{0}_features_control_vs_treated_wells_raw.csv".format(platename)),
        wells,
        platename
    )


if __name__ == "__main__":
    compareControlToTreatedWells(*sys.argv[1:])
