from __future__ import division
import pandas as pd
import numpy as np
import os, sys, csv


def aggregate(outputdir, metadatadir, resourcesdir):
    # get list of normalized files
    normalized_files = os.listdir(outputdir)
    normalized_files = [f for f in normalized_files if 'normalized' in f]

    # read normalized data into single df
    df_aggregated = pd.concat([pd.read_csv(os.path.join(outputdir, f)) for f in normalized_files])

    # add extra columns from selleck_plate_mappings
    selleck_plate_mappings = pd.read_csv(os.path.join(metadatadir, 'selleck_plate_mappings.csv'))
    df_aggregated = pd.merge(df_aggregated, selleck_plate_mappings, on='plate', how='left')

    # add compound annotations
    compound_annotations = pd.read_csv(os.path.join(resourcesdir, 'selleck_plates','Selleck_library_compound_annotations.csv'))
    df_aggregated_with_annots = pd.merge(df_aggregated, compound_annotations, on=['selleck_plate','well'], how='left')

    # save aggregated df
    df_aggregated.to_csv(os.path.join(outputdir, "aggregated_welldata.csv"), index=False)
    df_aggregated_with_annots.to_csv(os.path.join(outputdir, "aggregated_welldata_with_annots.csv"), index=False)

    # drop run column
    df_aggregated = df_aggregated.drop(['run'], axis=1)

    # group data by timepoints (6h, 24h), averaging the runs together
    df_grouped = df_aggregated.groupby(['selleck_plate', 'well', 'well_type', 'timepoint']).mean().reset_index()
    df_grouped_with_annots = pd.merge(df_grouped, compound_annotations, on=['selleck_plate', 'well'], how='left')

    # save grouped data
    df_grouped.to_csv(os.path.join(outputdir, "aggregated_welldata_grouped.csv"), index=False)
    df_grouped_with_annots.to_csv(os.path.join(outputdir, "aggregated_welldata_grouped_with_annots.csv"), index=False)


if __name__ == "__main__":
    aggregate(*sys.argv[1:])
