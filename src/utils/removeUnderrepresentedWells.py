import os
import sys
import pandas as pd
from src.utils import cxpPrinter


def removeUnderrepresentedWells(input_dir, output_dir, metadata_dir, drop_replicates=False):
    cxpPrinter.cxpPrint('Removing underrepresented wells')
    DROP_FLAG = 1

    # wells to discard for insufficient number of active objects to represent well activity
    wells_to_discard_file = os.path.join(output_dir, 'active_objects', 'wells_to_discard.csv')
    df_wells_to_discard = pd.read_csv(wells_to_discard_file)
    df_wells_to_discard['drop'] = DROP_FLAG  # flag for dropping

    # wells to discard for low correlation between replicates, i.e. between run 1 and run 2
    if drop_replicates:
        replicates_to_discard_file = os.path.join(metadata_dir, 'replicates_to_drop.csv')
        df_replicates_to_discard = pd.read_csv(replicates_to_discard_file)
        df_replicates_to_discard['drop'] = DROP_FLAG  # flag for dropping

    # read plate mapping from 'plate' <--> 'selleck_plate'
    mapping_cols_to_keep = ['selleck_plate', 'plate']
    selleck_plate_mappings = pd.read_csv(os.path.join(metadata_dir, 'selleck_plate_mappings.csv'))[mapping_cols_to_keep]

    files = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if '.csv' in f]
    for f in files:
        cxpPrinter.cxpPrint('Working on file ' + f)

        # read csv file as pandas dataframe
        df = pd.read_csv(f)

        # add 'selleck_plate' column to all rows from plate mapping
        df = pd.merge(df, selleck_plate_mappings, on='plate', how='left')

        # remove wells to discard based on active cell count
        df = df.merge(df_wells_to_discard, how='left', on=['plate', 'well'])
        df = df[df['drop'] != DROP_FLAG]
        df = df.drop(['drop'], axis=1)

        # remove wells to discard based on replicate correlation
        if drop_replicates:
            df = df.merge(df_replicates_to_discard, how='left', on=['selleck_plate', 'well'])
            df = df[df['drop'] != DROP_FLAG]
            df = df.drop(['drop'], axis=1)

        # remove 'selleck_plate' annotation
        df = df.drop(['selleck_plate'], axis=1)

        # overwrite original file
        df.to_csv(f, index=False)


if __name__ == "__main__":
    removeUnderrepresentedWells(*sys.argv[1:])
