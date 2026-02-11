# llama.cpp - Ubuntu CUDA and Vulkan binaries

This repo builds binary versions of `llama.cpp` for architectures that are not already part of the normal builds, such as Linux with CUDA or Vulkan support, and arm64 CPU or Vulkan.

New releases are automatically built for the latest release version of `llama.cpp`.

## CUDA

Currently supported CUDA build configurations:

| CPU arch   | OS           | CUDA  | Nvidia Compute arch |
|--------|--------------|-------|---------|
| amd64  | Ubuntu 24.04      | 12.9  | 86, 89  |
| amd64  | Ubuntu 24.04      | 13.0.88  | 86, 89  |
| arm64  | Ubuntu 22.04      | 12.9  | 87  |
| arm64  | Ubuntu 22.04      | 13.0.88  | 87  |

Compute architectures `86` and `89` are those used by consumer video cards.

Compute architecture `87` is used by Jetson Orin and Jetson AGX.

## Vulkan

Currently supported Vulkan build configurations:

| CPU arch   | OS           | Vulkan  |
|--------|--------------|-------|
| arm64  | Ubuntu 22.04 | 1.4.328.1  |

The prebuilt Vulkan SDK for ARM64 used for our builds comes from https://github.com/jakoch/vulkan-sdk-arm

Thank you!

## arm64 CPU

Currently supported CPU build configurations:

| CPU arch   | OS           |
|--------|--------------|
| arm64  | Ubuntu 22.04 |
