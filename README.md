# llama.cpp Ubuntu CUDA builds

This repo builds binary versions of `llama.cpp` for Ubuntu with CUDA support.

Currently supported build configurations:

| CPU arch   | OS           | CUDA  | Nvidia Compute arch |
|--------|--------------|-------|---------|
| amd64  | Ubuntu 24.04 | 12.9  | 86, 89  |
| arm64  | Ubuntu 22.04 | 12.9  | 87  |

Compute architectures `86` and `89` are those used by consumer video cards.

Compute architecture `87` is used by Jetson Orin and Jetson AGX.

Check releases for the latest version.
