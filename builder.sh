#!/usr/bin/env bash

export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

cd ./llama.cpp
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    -DLLAMA_CURL=OFF

cmake --build build --config Release
