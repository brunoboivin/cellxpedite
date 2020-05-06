#!/bin/bash
source ./util.sh

if [[ $OSTYPE == "darwin15" ]]; then
    shopt -s expand_aliases

    alias readlink="greadlink"

    alias md5sum="md5"
fi

progname=`basename $0`

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -b|--batchid)
    batch_id="$2"
    shift
    ;;
    --plate)
    plate_id="$2"
    shift
    ;;
    --filelist_dir)
    filelist_dir="$2"
    shift
    ;;
    --filelist_filename)
    filelist_filename="$2"
    shift
    ;;
    --cp_docker_image)
    cp_docker_image="$2"
    shift
    ;;
    --pipeline)
    pipeline_file="$2"
    shift
    ;;
    -t|--tmpdir)
    tmp_dir="$2"
    shift
    ;;
    -o|--outputdir)
    batchfile_dir="$2"
    shift
    ;;
    *)
    echo "unknown option"
    ;;
esac
shift
done

cp_docker_image="${cp_docker_image:-${cp_docker_image}}"

pathname_basename="${pathname_basename:-/home/ubuntu/bucket/}"

tmp_dir="${tmp_dir:-/tmp}"

groups_filename=cpgroups.csv

base_dir=../../docker

docker_cmd_filename=cp_docker_commands.txt

dcp_config_filename=dcp_config.json

s3_bucket=imaging-platform
#------------------------------------------------------

pipeline_file=`readlink -m ${pipeline_file}`

pipeline_filename=`basename ${pipeline_file}`

pipeline_tag=`echo ${pipeline_filename}|cut -d"." -f1`

pipeline_dir=`dirname ${pipeline_file}`

filelist_file=`readlink -m ${filelist_dir}/${filelist_filename}`


create_and_check_dir ${base_dir}/batchfiles > /dev/null


batchfile_dir=$(create_and_check_dir $batchfile_dir)

batchfile=${batchfile_dir}/Batch_data.h5


create_and_check_dir ${base_dir}/dockeranalysis > /dev/null

output_dir=`readlink -m ${base_dir}/dockeranalysis`/${pipeline_tag}
output_dir=$(create_and_check_dir $output_dir)

create_and_check_dir ${base_dir}/status > /dev/null

status_dir=`readlink -m ${base_dir}/status`/${pipeline_tag}

status_dir=$(create_and_check_dir $status_dir)

groups_file=${batchfile_dir}/${groups_filename}

docker_cmd_file=${batchfile_dir}/${docker_cmd_filename}

dcp_config_file=${batchfile_dir}/${dcp_config_filename}

echo --------------------------------------------------------------
echo pipeline_file       = ${pipeline_file}
echo base_dir            = ${base_dir}
echo tmp_dir             = ${tmp_dir}
echo pathname_basename   = ${pathname_basename}
echo filelist_file       = ${filelist_file}
echo batchfile_dir       = ${batchfile_dir}
echo batchfile           = ${batchfile}
echo output_dir          = ${output_dir}
echo groups_file         = ${groups_file}
echo docker_cmd_file     = ${docker_cmd_file}
echo dcp_config_file     = ${dcp_config_file}
echo --------------------------------------------------------------


if [[ -e $filelist_file ]]; then
    filelist_or_datafile="--file-list=/filelist_dir/${filelist_filename}"

else
    echo "Either filelist_file or datafile_file must be defined"

    exit 1

fi

check_path exists $pipeline_file

check_path exists $output_dir

check_path exists $batchfile_dir


echo "Creating batch file ${batchfile}"


check_path not_exists ${batchfile}

check_result=$?


if [[ $check_result = 0 || $check_result = 1 ]]; then
    rm -rf ${batchfile}

    docker run \
        --rm \
        --volume=${pipeline_dir}:/pipeline_dir \
        --volume=${filelist_dir}:/filelist_dir \
        --volume=${batchfile_dir}:/batchfile_dir \
        --volume=${tmp_dir}:/tmp_dir \
        --volume=${pathname_basename}:${pathname_basename} \
        ${cp_docker_image} \
        -p /pipeline_dir/${pipeline_filename} \
        ${filelist_or_datafile} \
        -o /batchfile_dir \
        -t /tmp_dir

fi

check_path exists ${batchfile}

echo "Creating groups file $groups_file"

check_cmd_exists aws

check_cmd_exists csvcut

check_cmd_exists jq

docker run \
    --rm \
    --volume=${pipeline_dir}:/pipeline_dir \
    --volume=${filelist_dir}:/filelist_dir \
    --volume=${batchfile_dir}:/batchfile_dir \
    --volume=${tmp_dir}:/tmp_dir \
    --volume=${pathname_basename}:${pathname_basename} \
    ${cp_docker_image} \
    --print-groups=/batchfile_dir/Batch_data.h5 |
    jq "map(.[0])" - | \
    in2csv -f json > ${groups_file}


function efs_s3_rel { 
    echo $(echo $1 | sed s,/home/ubuntu/efs,projects,1)
}

function efs_s3_abs { 
    echo $(echo $1 | sed s,/home/ubuntu/efs,s3://${s3_bucket}/projects,1)
}


ncols=`head -1 ${groups_file} | sed 's/[^,]//g' | wc -c`

group_name="`seq ${ncols}|parallel echo {{1}}-|tr -d "\n"|sed "s/-$//"`"

group_opts="`head -1 ${groups_file} | tr "," "\n" | nl | awk '{print $2 "={" $1 "}"}' | tr "\n" "," | sed 's/\,$//'`"

project_name=$(pwd|cut -d"/" -f5)

log_group_name=${project_name}_${batch_id}
echo "${log_group_name}"


echo "Creating docker commands file $docker_cmd_file"

parallel \
    --dry-run \
    -a ${batchfile_dir}/${groups_filename} \
    --header ".*\n" \
    -C "," \
    docker run \
    --rm \
    --volume=${pipeline_dir}:/pipeline_dir \
    --volume=${filelist_dir}:/filelist_dir \
    --volume=${batchfile_dir}:/batchfile_dir \
    --volume=${output_dir}/${group_name}:/output_dir \
    --volume=${tmp_dir}:/tmp_dir \
    --volume=${status_dir}:/status_dir \
    --volume=${pathname_basename}:${pathname_basename} \
    --log-driver=awslogs \
    --log-opt awslogs-group=${log_group_name} \
    --log-opt awslogs-stream=${group_name} \
    ${cp_docker_image} \
    -p /batchfile_dir/Batch_data.h5 \
    -g ${group_opts} \
    ${filelist_or_datafile} \
    -o /output_dir \
    -t /tmp_dir \
    -d /status_dir/${group_name}.txt > ${docker_cmd_file}


check_path exists ${docker_cmd_file}