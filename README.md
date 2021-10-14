# non-colocational-mps
Multiple point statistical algorithm that takes into account non-colocational uncertain (soft) data.

Uses some functions from mGstat (included) https://github.com/cultpenguin/mGstat

Written in MATLAB except for two CUDA C kernels that can be compiled for your specific hardware.

## Compiling cuda kernels
nvcc -arch=compute_61 -use_fast_math -ptx .\impalaFindSmem.cu

nvcc -arch=compute_61 -use_fast_math -ptx .\multiplyArray.cu