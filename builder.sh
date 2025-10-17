#!/usr/bin/env bash

export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

cd ./llama.cpp
cmake -B build \
    -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_CUDA=ON \
    -DGGML_BACKEND_DL=ON \

cmake --build build --config Release -j $(nproc)
