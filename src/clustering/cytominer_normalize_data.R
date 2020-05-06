# To clean, normalize, and potentially select representative subset of data.
# ** Use this file if code runs in separate instance with all files to 'cytomine'.
# Usage:  Rscript "input-folder" "filename-pattern" "output-folder"
#         Rscript cytominer_normalize_data.R "input" "*features_objects*" "output"


# import required packages
library(dplyr)
library(ggplot2)
library(magrittr)
library(stringr)
library(readr)

# function that replaces NaN values with ''
replaceNaNwithEmpty <- function(m){
  m[is.na(m)] <- ""
  return(m)
}

# process command line args
args = commandArgs(trailingOnly=TRUE)
input_dir <- args[1]
pattern <- args[2]
output_dir <- args[3]

# get list of input file names
celldata_filenames <- list.files(path = input_dir, pattern = pattern)

# for each input file, repeat the following steps:
#  - normalize data
#  - aggregated data
#  - select data
#  - save above data sets
for(celldata_filename in celldata_filenames){
  
  # construct input file path
  celldata_file <- file.path(input_dir, celldata_filename)

  # read cell data for entire plate
  object <- read_csv(celldata_file)

  # retrieve data columns; first 4 columns is metadata, other cols is cell data
  variables <- colnames(object)[-(1:4)]  # get all cols except first 4 (metadata)

  # define groupings (all metadata except cell id as we want to average this data)
  groupings <-
    c("plate",
      "well",
      "well_type"
    )
  # select cols of interest
  object %<>%
    dplyr::select(dplyr::one_of(c(groupings,variables)))

  # normalized data to neg control wells
  normalized <-
    cytominer::normalize(
      population = object %>%
        dplyr::collect(),
      variables = variables,
      strata =  c("plate"),
      sample =
        object %>%
        filter(well_type == "neg_control") %>%
        dplyr::collect()
    )
  normalized %<>% dplyr::collect()

  # aggregate data; this step yields the profiles
  aggregated <-
    cytominer::aggregate(
      population = normalized,
      variables = variables,
      strata = groupings
    ) %>%
    dplyr::collect()

  # extract analysis folder name from input filename
  analysis_folder_name =  sub("^([^.]*).*", "\\1", celldata_filename)
  strlocation = str_locate(analysis_folder_name, "_plate_features_objects")
  analysis_folder_name = substr(analysis_folder_name, 0, strlocation[1]-1)

  # save aggregated data
  output_file_aggregated = file.path(output_dir, paste(analysis_folder_name, "_welldata_normalized.csv", sep=""))
  write_csv(replaceNaNwithEmpty(aggregated), output_file_aggregated, append = FALSE, col_names = TRUE)


  # remove rows with NA values from aggregated
  aggregated_without_narows <- aggregated[complete.cases(aggregated),]

  # selected
  # - selected representative subset of columns from the sample space so here aggregated data
  #   without the rows containing NA values.
  # - then we retrieve these columns from the population space, here aggregated.
  # ** the implications of this are:
  #     - 'selected' might contain rows with NA values
  #     - subset of representative columns determined only from rows with some peaks
  selected <-
    cytominer::variable_select(
      population = aggregated,
      variables = variables,
      sample = aggregated_without_narows,
      operation = "correlation_threshold"
    ) %>%
    dplyr::collect()

  # save selected data
  output_file_selected = file.path(output_dir, paste(analysis_folder_name, "_welldata_selected.csv", sep=""))
  write_csv(replaceNaNwithEmpty(selected), output_file_selected, append = FALSE, col_names = TRUE)

  # save trimmed selected data, i.e. remove rows with NA values
  selected_without_narows <- selected[complete.cases(selected),]
  output_file_selected_nonan = file.path(output_dir, paste(analysis_folder_name, "_welldata_selected_nonan.csv", sep=""))
  write_csv(replaceNaNwithEmpty(selected_without_narows), output_file_selected_nonan, append = FALSE, col_names = TRUE)

}
