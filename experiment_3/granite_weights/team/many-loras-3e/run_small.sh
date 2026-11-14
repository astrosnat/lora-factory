#!/bin/bash
# Track "the denoised LoRA" (many-loras-3e): the dW / L / W analysis on a reduced layer set, on a CPU, weights only.
# Input: a folder made by fetch_small.py (a few layers of the non-holdout means, read from HF by range requests).
# Usage: ./run_small.sh FOLDER PYTHON "0,10,20,26,30,31" "q,k,v,o,dn" STEP...     steps: nh nhL nhW compare lensL lensW
D=$1; PY=$2; LAYERS=$3; TYPES=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=${THREADS:-2} OPENBLAS_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2} TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
N=960,960,480
mkdir -p $D/3e
for step in "$@"; do
  log=$D/3e/small_$step.log
  case $step in
    nh)      nice -n 10 $PY $HERE/mean_structure.py --ws $D --prefix nh_mean --tag nh --n_half $N --hold_from 60 --types $TYPES --device cpu > $log 2>&1 ;;
    nhL)     nice -n 10 $PY $HERE/mean_structure.py --ws $D --prefix nh_mean_L --tag nhL --n_half $N --hold_from 60 --energy e_parts.npy:1 --types $TYPES --device cpu > $log 2>&1 ;;
    nhW)     nice -n 10 $PY $HERE/mean_structure.py --ws $D --prefix nh_mean --minus nh_mean_L --tag nhW --n_half $N --hold_from 60 --energy e_parts.npy:0 --types $TYPES --device cpu > $log 2>&1 ;;
    compare) nice -n 10 $PY $HERE/compare_parts.py --dir3e $D/3e --tags dW=nh,W=nhW,L=nhL --types $TYPES > $log 2>&1 ;;
    lensL)   nice -n 10 $PY $HERE/lens.py --ws $D --tag nhL --layer_map $LAYERS --base_file $D/granite_small.safetensors --n_rand ${NRAND:-10} --device cpu > $log 2>&1 ;;
    lensW)   nice -n 10 $PY $HERE/lens.py --ws $D --tag nhW --layer_map $LAYERS --base_file $D/granite_small.safetensors --n_rand ${NRAND:-10} --device cpu > $log 2>&1 ;;
    lensD)   nice -n 10 $PY $HERE/lens.py --ws $D --tag nh --layer_map $LAYERS --base_file $D/granite_small.safetensors --n_rand ${NRAND:-10} --device cpu > $log 2>&1 ;;
  esac
  echo "$(date -u +%T) $step exit $? : $(tail -c 160 $log | tr '\n' ' ')" >> $D/3e/small_steps.txt
done
