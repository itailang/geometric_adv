#!/usr/bin/env bash

cd ./transfer/atlasnet/

# update paths
export CUDA_HOME=/usr/local/cuda-10.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin

# compile 3D-Chamfer Distance op
python auxiliary/ChamferDistancePytorch/chamfer3D/setup.py install
