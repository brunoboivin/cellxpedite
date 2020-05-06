function check_path {

    state=$1

    pathvar=$2

    if [[ ($state != "exists") && ($state != "not_exists") ]]; then 
    (>&2 echo "Unknown state: $state")

    exit 1
    fi

    tstamp=`date`

    if [[ $state == "exists" && ! -a "$pathvar" ]]; then
    message="not created / does not exist."

    (>&2 echo "[$tstamp] ${pathvar}" $message "Exiting.")
        
    exit 1
    fi
    
    if [[ $state == "not_exists" && -a "$pathvar" ]]; then
        message="exists."

        (>&2 echo "[$tstamp] ${pathvar} $message")

        while true; do
        read -p "Overwrite? (Y/N)" yn
        case $yn in
            [Yy]* ) return 1 ;;
            [Nn]* ) return 2 ;;
            * ) echo "Please answer yes or no.";;
        esac
        done

    fi

    return 0
}

function info {
    tstamp=`date`

    (>&2 echo "[$tstamp] $1")
}

function compare_md5() {

    A=$1

    B=$2

    MD5_A=`md5sum ${A} | cut -f1 -d' '`

    MD5_B=`md5sum ${B} | cut -f1 -d' '`
    
    if [[ $MD5_A != $MD5_B ]]; then
    
    tstamp=`date`
    
    (>&2 echo "[$tstamp] md5 check failed: ${A} and ${B} are different")

    exit 1

    fi
}

function create_and_check_dir () {

    dname=$1

    mkdir -p ${dname}

    check_path exists ${dname}
    
    echo `readlink -e $dname`
}


function move_and_check_file () {

    src=$1

    dst=$2

    info "Moving ${src} to ${dst}"

    rsync -a ${src} ${dst}

    compare_md5 ${src} ${dst}

    rm ${src}

}

function check_cmd_exists () {
    # This doesn't catch the case where the cmd is a python command that is 
    # available in a different pyenv environment.

    cmdstr=$1

    eval type $cmdstr >/dev/null 2>&1 || { 

    echo >&2 "$cmdstr not installed.  Aborting."

    exit 1
    }

}
