from __future__ import division
import pandas as pd
import numpy as np
import sys, os
from src.utils import metadataExtractor, cxpPrinter


def createPlateObjectsFile(dataframes,output_filepath):
    df = pd.concat(dataframes)

    # reorder df_plate columns based on df that has the most columns
    max_numcol_ind = np.asscalar(np.argmax([len(list(d.columns)) for d in dataframes]))
    df = df[list(dataframes[max_numcol_ind].columns)]

    # save all object features
    df.to_csv(output_filepath, index=False)
    return df


def createPlateFeaturesSummary(df,columns,output_filepath):
    df_grouped = df.groupby(['plate','well','well_type'])
    df = df_grouped.mean().join(pd.DataFrame(df_grouped.size(),columns=['count'])).reset_index()
    df['obj_number'] = df['count']
    df = df.rename(columns={'obj_number': 'number_of_objects'})
    df = df.drop(['count'], axis=1)

    # add total peak count columns
    df_sum = df_grouped[columns].sum().reset_index()
    new_colnames_mapping = dict(zip(columns,[c+'_total' for c in columns]))
    df_sum = df_sum.rename(columns=new_colnames_mapping)
    df = pd.merge(df, df_sum)

    # save df to csv file
    df.to_csv(output_filepath, index=False)


def mergeWellFeatures(config, threshold_multiplier=1.0):
    cxpPrinter.cxpPrint("Merging well features")

    # get paths and metadata
    metadata_dict = metadataExtractor.import_metadata(config)
    outputdir = metadata_dict["config"]["var"]["outputdir"]
    wellnamespath = metadata_dict["config"]["var"]["wellnamespath"]
    platename = metadata_dict["config"]["var"]["platename"]

    # update output dir if threshold multiplier is provided
    threshold_multiplier = float(threshold_multiplier)
    if threshold_multiplier != 1.0:
        outputdir = os.path.join(outputdir, 't'+str(threshold_multiplier))

    # get list of wells
    text_file = open(wellnamespath, "r")
    wells = text_file.readlines()
    wells = [well.replace('\n', '') for well in wells]
    text_file.close()

    # get list of dataframes and merge them into a single frame
    dataframes_norm = [pd.read_csv(os.path.join(outputdir,"{0}_features.csv".format(well))) for well in wells]
    df_plate_objects_norm = createPlateObjectsFile(dataframes_norm,os.path.join(outputdir, "{0}_plate_features_objects.csv".format(platename)))
    columns_norm = ['WM_peak_count', 'SM_peak_count', 'LM_peak_count']
    createPlateFeaturesSummary(df_plate_objects_norm,columns_norm,os.path.join(outputdir, "{0}_plate_features_wells.csv".format(platename)))

    dataframes_raw = [pd.read_csv(os.path.join(outputdir, "{0}_features_raw.csv".format(well))) for well in wells]
    df_plate_objects_raw = createPlateObjectsFile(dataframes_raw, os.path.join(outputdir,"{0}_plate_features_objects_raw.csv".format(platename)))
    columns_raw = ['RAW_WM_peak_count', 'RAW_SM_peak_count', 'RAW_LM_peak_count']
    createPlateFeaturesSummary(df_plate_objects_raw, columns_raw,os.path.join(outputdir, "{0}_plate_features_wells_raw.csv".format(platename)))


if __name__ == "__main__":
    mergeWellFeatures(*sys.argv[1:])
