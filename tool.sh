#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
TEMP=$(getopt -o '' --long train,pyrun:,display: -- "$@")
if [ $? != 0 ] ; then echo "Terminating ..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
USER_DISPLAY=${DISPLAY}

train(){
echo train
}

pyrun(){
    python3 ${SCRIPT_DIR}/$1
}
while true ; do
    case "$1" in 
        --train)
            MODEL="TRAIN"
            shift 
            ;;
        --pyrun)
            MODEL="PYRUN"
            PYRUN_PATH="$2"
            shift 2 
        ;;
        --display)
            USER_DISPLAY="$2"
            shift 2 
        ;;
        --)
            shift 
            break
            ;;
        *)
            echo "error"
            exit 1
            ;;
    esac
done

export DISPLAY=${USER_DISPLAY}
case "${MODEL}" in
    TRAIN)
        train
        ;;
    PYRUN)
        pyrun "${PYRUN_PATH}"
        ;;
esac
