#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load cuda
nvcc -lineinfo -arch=sm_80 -O3 src/fft.cu src/fft_optimization.cu performance.cu -o test -lcufft
ncu -f -o firstrun --set full --import-source yes ./test