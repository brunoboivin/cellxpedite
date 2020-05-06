#!/bin/bash
# give EC2 instance time to initialize
sleep 30
source /home/ubuntu/.profile

# Variables
DATASET=$1

# for validation screen
S3INPUTBUCKET="hms-woolf-kasper/analyses/Kasper/validation_screen"
S3OUTPUTBUCKET="hms-woolf-kasper/analyses/Kasper/validation_screen/clustering"

GITHUBPROJECT=$4
GITPROJECTNAME=$(echo $GITHUBPROJECT | sed 's/.*\/\(.*\).git/\1/g')
GITPROJECTPATH="/home/ubuntu/${GITPROJECTNAME}"
NUMJOBSPARALLEL=$5
ANALYSISTYPE=$6
PROJECTPATH="/home/ubuntu/project"
INPUTDIR="${PROJECTPATH}/input"
OUTPUTDIR="${PROJECTPATH}/output"
METADATADIR="${PROJECTPATH}/metadata"
SRCPATH="/home/ubuntu/${GITPROJECTNAME}/src"
RESOURCESDIR="/home/ubuntu/${GITPROJECTNAME}/src/res"
CONFIGFILE="${SRCPATH}/res/config/config.cfg"

# Setup project folders
mkdir -p "${INPUTDIR}"
mkdir -p "${OUTPUTDIR}"
mkdir -p "${METADATADIR}"

# Clone project source code from GitHub
chmod 400 ~/.ssh/id_rsa
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
git clone ${GITHUBPROJECT}

# add platename and analysis type to config file
cd /home/ubuntu/${GITPROJECTNAME}/src
echo "" >> ${CONFIGFILE}
echo "resourcesdir: ${SRCPATH}/res" >> ${CONFIGFILE}
echo "platename: ${DATASET}" >> ${CONFIGFILE}
echo "analysistype: ${ANALYSISTYPE}" >> ${CONFIGFILE}

# set python path to root of project
# ** Necessary step to avoid Python import issues
PYTHONPATH=$PYTHONPATH:${GITPROJECTPATH}
export PYTHONPATH

source activate cxp

# declare array of analysis folders to fetch
declare -a analysis_folders=(
    "6h/R44821_190620160001"
    "24h/R44821_190621160001"
)

# save analysis folders info
printf "%s\n" "${analysis_folders[@]}" > "${METADATADIR}/analysis_folders_list.txt"

# copy mapping to metadata folder
cp ${SRCPATH}/res/selleck_plates/selleck_plate_mappings.csv "${METADATADIR}"

# copy replicates data to metadata folder
aws s3 cp "s3://${S3INPUTBUCKET}/replicates/" "${METADATADIR}" --recursive

# repeat aggregation/clustering for all threshold folders specified
threshold_multipliers=(-999999.9 0.5 1.0 2.0)
for tm in "${threshold_multipliers[@]}"
do
    # create output subdirectories
    mkdir -p "${INPUTDIR}/t$tm"
    mkdir -p "${OUTPUTDIR}/t$tm"

    # Step 1: Copy needed files from S3
    # get files from clustering output folder (if existent)
    aws s3 sync "s3://${S3OUTPUTBUCKET}" "${PROJECTPATH}"
    # fetch input files from analysis folders if needed
    for folder_path in "${analysis_folders[@]}"
    do
       folder_name=$(basename "${folder_path}")
       if (( $(echo "$tm == 1.0" | bc -l) )); then
            aws s3 cp "s3://${S3INPUTBUCKET}/${folder_path}/output/${folder_name}_plate_features_objects.csv" "${INPUTDIR}/t$tm"
       else
            aws s3 cp "s3://${S3INPUTBUCKET}/${folder_path}/output/t$tm/${folder_name}_plate_features_objects.csv" "${INPUTDIR}/t$tm"
       fi
    done

    # Step 2: Aggregate data
    if [ "$(ls -A ${INPUTDIR}/t$tm)" ]; then
        cd /home/ubuntu/${GITPROJECTNAME}/src
        # fill na values with zeros
        python ${SRCPATH}/utils/fillWithZeros.py "${INPUTDIR}"/t$tm
        # identify active cells
        python ${SRCPATH}/analysis/identifyActiveCells.py "${INPUTDIR}"/t$tm "${OUTPUTDIR}/t$tm" "${METADATADIR}"
        # remove underrepresented wells
        python ${SRCPATH}/utils/removeUnderrepresentedWells.py "${INPUTDIR}"/t$tm "${OUTPUTDIR}/t$tm" "${METADATADIR}"
        # use cytominer to normalize data
        Rscript ${SRCPATH}/clustering/cytominer_normalize_data.R "${INPUTDIR}"/t$tm "*features_objects*" "${OUTPUTDIR}/t$tm"
        # aggregate normalized data into single file
        python ${SRCPATH}/clustering/aggregate.py "${OUTPUTDIR}/t$tm" "${METADATADIR}" "${RESOURCESDIR}"
    fi
done

# Step 4: Copy data from instance to S3
aws s3 sync "${PROJECTPATH}" "s3://${S3OUTPUTBUCKET}"

# Stop instance
sudo shutdown -h now

echo "*** Successfully Completed ***"
