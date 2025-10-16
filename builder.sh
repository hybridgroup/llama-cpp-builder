#!/usr/bin/env bash

cd ./llama.cpp
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    -DLLAMA_CURL=OFF

cmake --build build --config Release
