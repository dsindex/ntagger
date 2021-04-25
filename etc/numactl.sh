#!/bin/bash

set -o nounset
set -o errexit

VERBOSE_MODE=0

function error_handler()
{
  local STATUS=${1:-1}
  [ ${VERBOSE_MODE} == 0 ] && exit ${STATUS}
  echo "Exits abnormally at line "`caller 0`
  exit ${STATUS}
}
trap "error_handler" ERR

PROGNAME=`basename ${BASH_SOURCE}`
DRY_RUN_MODE=0

function print_usage_and_exit()
{
  set +x
  local STATUS=$1
  echo "Usage: ${PROGNAME} [-v] [-v] [--dry-run] [-h] [--help]"
  echo ""
  echo " Options -"
  echo "  -v                 enables verbose mode 1"
  echo "  -v -v              enables verbose mode 2"
  echo "      --dry-run      show what would have been dumped"
  echo "  -h, --help         shows this help message"
  exit ${STATUS:-0}
}

function debug()
{
  if [ "$VERBOSE_MODE" != 0 ]; then
    echo $@
  fi
}

GETOPT=`getopt -o vh --long dry-run,help -n "${PROGNAME}" -- "$@"`
if [ $? != 0 ] ; then print_usage_and_exit 1; fi

eval set -- "${GETOPT}"

while true
do case "$1" in
     -v)            let VERBOSE_MODE+=1; shift;;
     --dry-run)     DRY_RUN_MODE=1; shift;;
     -h|--help)     print_usage_and_exit 0;;
     --)            shift; break;;
     *) echo "Internal error!"; exit 1;;
   esac
done

if (( VERBOSE_MODE > 1 )); then
  set -x
fi


# template area is ended.
# -----------------------------------------------------------------------------
if [ ${#} != 0 ]; then print_usage_and_exit 1; fi

# current dir of this script
CDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]})))
PDIR=$(readlink -f $(dirname $(readlink -f ${BASH_SOURCE[0]}))/..)

# -----------------------------------------------------------------------------
# functions

function make_calmness()
{
    exec 3>&2 # save 2 to 3
    exec 2> /dev/null
}

function revert_calmness()
{
    exec 2>&3 # restore 2 from previous saved 3(originally 2)
}

function close_fd()
{
    exec 3>&-
}

function jumpto
{
    label=$1
    cmd=$(sed -n "/$label:/{:a;n;p;ba};" $0 | grep -v ':$')
    eval "$cmd"
    exit
}


# end functions
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# main 

make_calmness
if (( VERBOSE_MODE > 1 )); then
    revert_calmness
fi

# reference : https://huggingface.co/blog/bert-cpu-scaling-part-1

numactl --hardware
numactl --show
lscpu

# Architecture:        x86_64
# CPU op-mode(s):      32-bit, 64-bit
# Byte Order:          Little Endian
# CPU(s):              56
# On-line CPU(s) list: 0-55
# Thread(s) per core:  2
# Core(s) per socket:  14
# Socket(s):           2
# NUMA node(s):        2

numactl -C 0-14   -m 0 python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-kmou-morph.pt --data_dir=data/kmou2019 --bert_output_dir=bert-checkpoint-kor-kmou-morph --use_crf --bert_use_pos --device=cpu --num_examples=100 --num_threads=14 
numactl -C 15-27  -m 1 python evaluate.py --config=configs/config-bert.json --model_path=pytorch-model-bert-kor-kmou-morph.pt --data_dir=data/kmou2019 --bert_output_dir=bert-checkpoint-kor-kmou-morph --use_crf --bert_use_pos --device=cpu --num_examples=100 --num_threads=14 

close_fd

# end main
# -----------------------------------------------------------------------------
