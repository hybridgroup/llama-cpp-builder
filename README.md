# llama.cpp Ubuntu CUDA builds

This repo builds binary versions of `llama.cpp` for Ubuntu 24.04 with CUDA support.

It only supports Ubuntu 24.04 with the NVidia 12.9 drivers. Furthermore, the only CUDA architectures in the prebuilt binaries are `86` and `89`.

Since these are the architectures used by consumer video cards, this should be sufficient for use as part of a local environment.

Check releases for the latest version.

## building docker

```
docker build -t llama-cuda-builder .
```

## clone llama.cpp

Close the `llama.cpp` repo into a subdirectory.

## running

```
docker run --rm -v $(pwd):/src -a stdout -a stderr llama-cuda-builder
```

The build artifacts are in `llama.cpp/build/bin`
