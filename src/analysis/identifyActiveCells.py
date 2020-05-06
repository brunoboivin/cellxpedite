import os
import sys
import pandas as pd
import collections
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from src.utils import cxpPrinter


def plotDistribution(data, output_dir, title_prefix='', title='Number of active cells'):
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    # overall figure
    fig.suptitle("[{0}]  {1}".format(title_prefix, title))
    # histogram
    axes[0].hist(data, bins='auto')
    axes[0].set_xlabel('# active cells')
    axes[0].set_ylabel('# wells')
    # box plot
    axes[1].boxplot(data)
    axes[1].set_ylabel('# active cells')
    # save plot
    output_filename = "{0}_active_objects_distribution.png".format(title_prefix)
    plt.savefig(os.path.join(output_dir, output_filename))
    # close fig
    plt.close()


def plotIndividualDistribution(data, plate='', timepoint='', title='Number of active cells', title_prefix='', output_dir=None, bins='auto'):
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    # overall figure
    fig.suptitle("[{0},{1},{2}]  {3}".format(plate,timepoint,title_prefix,title))
    # histogram
    axes[0].hist(data, bins=bins)
    axes[0].set_xlabel('# active cells')
    axes[0].set_ylabel('# wells')
    # box plot
    axes[1].boxplot(data)
    axes[1].set_ylabel('# active cells')
    # save plot
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir,"{0}_{1}_{2}_active_objects_distribution.png".format(plate,timepoint,title_prefix)))
    # close fig
    plt.close()


def identifyActiveCells(input_dir, output_dir, metadata_dir):
    cxpPrinter.cxpPrint('Identifying active cells')
    cols_of_interest = [
        'plate', 'well', 'well_type', 'obj_number',
        'WM_peak_count', 'SM_peak_count', 'LM_peak_count',
        'wavelet8_peak_count', 'wavelet4_peak_count'
    ]

    # list of features dict
    d_list = []

    files = [os.path.join(input_dir,f) for f in os.listdir(input_dir) if '.csv' in f]
    for f in files:
        cxpPrinter.cxpPrint('Working on file ' + f)

        # read csv file as pandas dataframe
        df = pd.read_csv(f)

        # retain only cols of interest
        df = df[cols_of_interest]

        # count number of active cells for each well
        plate = df['plate'].unique()[0]
        wells = df['well'].unique()

        for well in wells:
            d_tmp = collections.OrderedDict()
            d_tmp['plate'] = plate
            d_tmp['well'] = well

            # get well data
            df_single_well = df[df['well'] == well]

            # compute total number of cells and total number of active cells for each method
            d_tmp['object_count'] = len(df_single_well)
            d_tmp['WM_active'] = len(df_single_well[df_single_well['WM_peak_count'] > 0])
            d_tmp['SM_active'] = len(df_single_well[df_single_well['SM_peak_count'] > 0])
            d_tmp['LM_active'] = len(df_single_well[df_single_well['LM_peak_count'] > 0])
            d_tmp['all_active'] = len(
                df_single_well[
                    (df_single_well['WM_peak_count'] > 0) &
                    (df_single_well['SM_peak_count'] > 0) &
                    (df_single_well['LM_peak_count'] > 0)
                    ]
            )
            d_tmp['any_active'] = len(
                df_single_well[
                    (df_single_well['WM_peak_count'] > 0) |
                    (df_single_well['SM_peak_count'] > 0) |
                    (df_single_well['LM_peak_count'] > 0)
                    ]
            )

            d_list.append(d_tmp)

    # construct df from dictionary
    df_active_cells = pd.DataFrame(d_list)

    # add extra columns from selleck_plate_mappings
    selleck_plate_mappings = pd.read_csv(os.path.join(metadata_dir, 'selleck_plate_mappings.csv'))
    # merge metadata into active cell df
    df_active_cells = pd.merge(df_active_cells, selleck_plate_mappings, on='plate', how='left')

    # save df of active objects
    output_filename = 'active_objects.csv'
    active_object_dir = os.path.join(output_dir, 'active_objects')
    if not os.path.exists(active_object_dir):
        os.makedirs(active_object_dir)
    output_file = os.path.join(active_object_dir, output_filename)
    df_active_cells.to_csv(output_file, index=False)

    # save list of wells to discard, i.e. wells with insufficient number of active cells to represent well activity
    num_active_objects_threshold = 20
    df_wells_to_discard = df_active_cells[df_active_cells['all_active'] <= num_active_objects_threshold][['plate','well']]
    df_wells_to_discard.to_csv(os.path.join(active_object_dir,'wells_to_discard.csv'), index=False)

    # create distributions subdir for storing plots
    distributions_dir = os.path.join(active_object_dir, 'distributions')
    if not os.path.exists(distributions_dir):
        os.makedirs(distributions_dir)

    # plot active object count distribution
    cxpPrinter.cxpPrint('Plotting aggregated distributions')
    prefixes = ['WM', 'SM', 'LM', 'all', 'any']
    for prefix in prefixes:
        plotDistribution(df_active_cells[prefix+'_active'], distributions_dir, title_prefix=prefix)

    # plot individual distributions
    cxpPrinter.cxpPrint('Plotting individual distributions')
    df_grouped = df_active_cells.groupby(['selleck_plate', 'timepoint'])
    for group_name, group_data in df_grouped:
        for prefix in prefixes:
            plate = group_name[0]
            timepoint = group_name[1]
            data = group_data[prefix + '_active'].tolist()
            data = [d for d in data if d != 0]  # remove wells with 0 active cells
            bins = 30
            plotIndividualDistribution(
                data=data,
                plate=plate,
                timepoint=timepoint,
                title_prefix=prefix,
                output_dir=distributions_dir,
                bins=bins
            )


if __name__ == "__main__":
    identifyActiveCells(*sys.argv[1:])
