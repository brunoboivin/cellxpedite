#!/bin/bash
# give EC2 instance time to initialize
sleep 30
source /home/ubuntu/.profile

# Variables
DATASET=$1
S3INPUTBUCKET=$2
S3OUTPUTBUCKET=$3
GITHUBPROJECT=$4
GITPROJECTNAME=$(echo $GITHUBPROJECT | sed 's/.*\/\(.*\).git/\1/g')
NUMJOBSPARALLEL=$5
ANALYSISTYPE=$6
ILLUMPIPELINE=$7
SEGMENTPIPELINE=$8
PROJECT="myplate"
PROJECTPATH="/home/ubuntu/bucket/projects/${PROJECT}"
ANALYSISPATH="${PROJECTPATH}/analysis"
TIMESTAMPSFILE="${ANALYSISPATH}/timestamps.txt"
INPUTDIR="/home/ubuntu/bucket/projects/${PROJECT}/input"
GITPROJECTPATH="/home/ubuntu/${GITPROJECTNAME}"
SRCPATH="${GITPROJECTPATH}/src"
CONFIGFILE="${SRCPATH}/res/config/config.cfg"

# Setup project folders
mkdir -p "${PROJECTPATH}"/input
mkdir -p "${ANALYSISPATH}"/metadata
mkdir -p "${ANALYSISPATH}"/illum
mkdir -p "${ANALYSISPATH}"/globdecay
mkdir -p "${ANALYSISPATH}"/maxproj
mkdir -p "${ANALYSISPATH}"/segments
mkdir -p "${ANALYSISPATH}"/filelists
mkdir -p "${ANALYSISPATH}"/timeseries
mkdir -p "${ANALYSISPATH}"/figures
mkdir -p "${ANALYSISPATH}"/output
mkdir -p /home/ubuntu/tmp

echo `date` "=> Start time" >> "${TIMESTAMPSFILE}"

# Copy image files from S3 to instance
source activate cxp
aws logs create-log-group --log-group-name "instance_"
aws s3 sync "s3://${S3OUTPUTBUCKET}/${DATASET}" "${ANALYSISPATH}"
#if [ ! "$(ls -A ${ANALYSISPATH}/timeseries)" ]; then
     aws s3 cp "s3://${S3INPUTBUCKET}/${DATASET}" "${INPUTDIR}" --recursive
#fi
echo `date` "=> CP S3->Instance" >> "${TIMESTAMPSFILE}"

# Clone project source code from GitHub
chmod 400 ~/.ssh/id_rsa
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
git clone ${GITHUBPROJECT}

# append info to config file
echo "" >> ${CONFIGFILE}
echo "resourcesdir: ${SRCPATH}/res" >> ${CONFIGFILE}
echo "platename: ${DATASET}" >> ${CONFIGFILE}
echo "analysistype: ${ANALYSISTYPE}" >> ${CONFIGFILE}

# set python path to root of project
# ** Necessary step to avoid Python import issues
PYTHONPATH=$PYTHONPATH:${GITPROJECTPATH}
export PYTHONPATH
cd ${GITPROJECTPATH}

# Initialize metadata files
if [ ! "$(ls -A ${ANALYSISPATH}/metadata)" ]; then
    # Remove compromised wells
    python ${SRCPATH}/utils/removeCompromisedWells.py ${CONFIGFILE}
    # Initialize metadata files
     python ${SRCPATH}/utils/metadataExtractor.py ${CONFIGFILE}
     # Remove inconsistent wells
     python ${SRCPATH}/utils/removeInconsistentWells.py ${CONFIGFILE}
fi
# Generate illum filelist
python ${SRCPATH}/utils/filelistGenerator.py "${INPUTDIR}" "${ANALYSISPATH}"/filelists ${CONFIGFILE} 0.2
source deactivate cxp

# Illumination correction pipeline
if [ ! "$(ls -A ${ANALYSISPATH}/illum)" ]; then
    source activate cellpainting_python_3
    chmod 777 ${GITPROJECTPATH}/instance/scripts/run_cell_profiler.sh
    cd ${GITPROJECTPATH}/instance/scripts
    ./run_cell_profiler.sh -b "${DATASET}" \
    --filelist_dir "${ANALYSISPATH}"/filelists \
    --filelist_filename illum_filelist.txt \
    --pipeline "../../pipelines/${ILLUMPIPELINE}" \
    --cp_docker_image woolflabhms/cellprofiler:2.2.1 \
    --tmpdir /home/ubuntu/tmp/ \
    -o "${ANALYSISPATH}"/illum
    source deactivate cellpainting_python_3
    echo `date` "=> Illum pipeline" >> "${TIMESTAMPSFILE}"
fi

# als env for python code
source activate cxp
cd ${GITPROJECTPATH}

# Global decay
if [ ! "$(ls -A ${ANALYSISPATH}/globdecay)" ]; then
    python ${SRCPATH}/img_processing/calculateGlobalDecay.py ${CONFIGFILE}
    echo `date` "=> Glob decay" >> "${TIMESTAMPSFILE}"
fi
# Max projections
if [ ! "$(ls -A ${ANALYSISPATH}/maxproj)" ]; then
    cat "${ANALYSISPATH}"/metadata/well_names.txt | parallel --no-notice -j ${NUMJOBSPARALLEL} python ${SRCPATH}/img_processing/computeMaxProjections.py ${CONFIGFILE} {}
    echo `date` "=> Max proj" >> "${TIMESTAMPSFILE}"
fi
source deactivate cxp

# Image segmentation based on max projections
if [ ! "$(ls -A ${ANALYSISPATH}/segments)" ]; then
    # Make segment filelist of images for Docker command
    find "${ANALYSISPATH}"/maxproj -type f > "${ANALYSISPATH}"/filelists/segment_filelist.txt

    # Segmentation pipeline ( ${SEGMENTPIPELINE} )
    source activate cellpainting_python_3
    chmod 777 ${GITPROJECTPATH}/instance/scripts/run_cell_profiler.sh
    cd ${GITPROJECTPATH}/instance/scripts
    ./run_cell_profiler.sh -b "${DATASET}" \
    --filelist_dir "${ANALYSISPATH}"/filelists \
    --filelist_filename segment_filelist.txt \
    --pipeline "../../pipelines/${SEGMENTPIPELINE}" \
    --cp_docker_image woolflabhms/cellprofiler:3.0.0 \
    --tmpdir /home/ubuntu/tmp/ \
    -o "${ANALYSISPATH}"/segments
    source deactivate cellpainting_python_3
    echo `date` "=> Segment pipeline" >> "${TIMESTAMPSFILE}"
fi

# Time series extraction & merging of fragments
#  ** we need the input images for this operation
source activate cxp
# remove substandard wells to prevent further analysis on them
python ${SRCPATH}/utils/removeSubstandardWells.py ${CONFIGFILE}

cd ${GITPROJECTPATH}
if [ ! "$(ls -A ${ANALYSISPATH}/timeseries)" ]; then
    # extract time series
    cat "${ANALYSISPATH}"/metadata/well_names.txt | parallel --no-notice -j ${NUMJOBSPARALLEL} python ${SRCPATH}/analysis/extractTimeseriesFromWell.py ${CONFIGFILE} {}
    echo `date` "=> Time series" >> "${TIMESTAMPSFILE}"

    # downsize fragment metadata to reduce computing time
    # permission change required for downsizing to be able to save downsized file
    sudo chown -R ubuntu:ubuntu ${ANALYSISPATH}/segments
    python ${SRCPATH}/analysis/downsizeFragmentFeatures.py ${CONFIGFILE}

    # merge object segments
    cat "${ANALYSISPATH}"/metadata/well_names.txt | parallel --no-notice -j ${NUMJOBSPARALLEL} python -W ignore ${SRCPATH}/img_processing/mergeObjectSegments.py ${CONFIGFILE} {}
    # rextract time series
    cat "${ANALYSISPATH}"/metadata/well_names.txt | parallel --no-notice -j ${NUMJOBSPARALLEL} python ${SRCPATH}/analysis/extractTimeseriesFromWell.py ${CONFIGFILE} {} "{0}_merged_neuronFragments.tiff"
fi

 # compute plate global extrema
 python ${SRCPATH}/analysis/computeGlobalExtrema.py ${CONFIGFILE}


## Feature extraction
# define threshold multipliers
# for each multiplier:
#   - create subfolder in output dir
#   - save extracted features accordingly

threshold_multipliers=(-999999.9 0.5 1.0 2.0)
for tm in "${threshold_multipliers[@]}"
do
mkdir -p "${ANALYSISPATH}/output/t$tm"
if [ "${ANALYSISTYPE}" == "standard" ]; then
    python ${SRCPATH}/analysis/getPeakThreshold.py ${CONFIGFILE} "standard"
    cat "${ANALYSISPATH}"/metadata/well_names.txt | parallel --no-notice -j ${NUMJOBSPARALLEL} python ${SRCPATH}/analysis/extractFeaturesFromWell.py ${CONFIGFILE} {} $tm
    python ${SRCPATH}/analysis/mergeWellFeatures.py ${CONFIGFILE} $tm
    python ${SRCPATH}/analysis/computeZfactors.py ${CONFIGFILE} $tm
    python ${SRCPATH}/comparisons/compareControlToTreatedWells.py ${CONFIGFILE} $tm
else
    cp ${SRCPATH}/res/threshold/peak_threshold.csv "${ANALYSISPATH}"/output/
    cat "${ANALYSISPATH}"/metadata/well_names.txt | parallel --no-notice -j ${NUMJOBSPARALLEL} python ${SRCPATH}/analysis/extractFeaturesFromWell.py ${CONFIGFILE} {} $tm
    python ${SRCPATH}/analysis/mergeWellFeatures.py ${CONFIGFILE} $tm
    python ${SRCPATH}/analysis/computeZfactors.py ${CONFIGFILE} $tm
    python ${SRCPATH}/comparisons/compareControlToTreatedWells.py ${CONFIGFILE} $tm
fi
echo `date` "=> Features" >> "${TIMESTAMPSFILE}"
done


# Copy analysis folder from instance to S3
aws s3 sync "${ANALYSISPATH}" "s3://${S3OUTPUTBUCKET}/${DATASET}"

# Stop instance
sudo shutdown -h now

echo "*** Successfully Completed ***"