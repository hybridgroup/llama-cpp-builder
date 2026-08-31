# llama.cpp Builder

## Prebuilt binaries for the platforms that the normal builds leave out

This repo builds binary versions of `llama.cpp` libraries and executables for architectures that are not already part of the normal builds: Linux with CUDA or Vulkan, Linux arm64 with CPU, Vulkan, or OpenCL, and WebAssembly for a browser.

New releases are automatically built for the latest release version of `llama.cpp`. The latest release is checked once per hour.

[![yzma logo](https://raw.githubusercontent.com/hybridgroup/yzma/refs/heads/main/images/yzma-logo-full-color-small.png)](https://github.com/hybridgroup/yzma)

Used by [yzma](https://github.com/hybridgroup/yzma) installer. yzma lets you write Go applications that directly integrate the latest `llama.cpp` libraries.

## CUDA

Currently supported CUDA build configurations:

| CPU arch   | OS           | CUDA  | Nvidia Compute arch |
|--------|--------------|-------|---------|
| amd64  | Ubuntu 24.04      | 12.9  | 75, 80, 86, 89, 90  |
| amd64  | Ubuntu 24.04      | 13.0.88  | 75, 80, 86, 89, 90  |
| arm64  | Ubuntu 22.04      | 12.9  | 87, 121  |
| arm64  | Ubuntu 22.04      | 13.0.88  | 87, 121  |

Compute architectures `86` and `89` are those used by consumer video cards.

Compute architecture `87` is used by Jetson Orin and Jetson AGX.

Compute architecture `121` is used by the GB10 superchip in the NVIDIA DGX Spark.

## Vulkan

Currently supported Vulkan build configurations:

| CPU arch   | OS           | Vulkan  |
|--------|--------------|-------|
| arm64  | Ubuntu 22.04/Debian Bookworm | 1.4.335.0  |
| arm64  | Ubuntu 24.04/Debian Trixie | 1.4.335.0  |

The prebuilt Vulkan SDK for ARM64 used for our builds comes from https://github.com/jakoch/vulkan-sdk-arm

Thank you!

## OpenCL

Currently supported OpenCL build configurations:

| CPU arch   | OS           |
|--------|--------------|
| arm64  | Ubuntu 24.04/Debian Trixie |

This is the backend that `llama.cpp` documents for Qualcomm Adreno GPUs, which is what an arm64 board with an Adreno has instead of CUDA or Vulkan.

The build needs the OpenCL headers and an OpenCL loader, which come from the `ocl-icd-opencl-dev`, `opencl-headers`, and `opencl-clhpp-headers` packages. The machine that runs the libraries needs a driver from the vendor of its GPU.

## CPU

Currently supported CPU build configurations:

| CPU arch   | OS           |
|--------|--------------|
| arm64  | Ubuntu 22.04/Debian Bookworm |
| arm64  | Ubuntu 24.04/Debian Trixie |

## WebAssembly

Currently supported WebAssembly build configurations:

| Variant | Asset | What a browser needs for it |
|--------|--------------|-------|
| One thread  | `llama-<tag>-bin-wasm-simd.tar.gz` | Nothing. It works everywhere. |
| More threads  | `llama-<tag>-bin-wasm-simd-mt.tar.gz` | `SharedArrayBuffer`, so a page with the COOP and COEP headers |
| WebGPU  | `llama-<tag>-bin-wasm-webgpu.tar.gz` | WebGPU with f16 shaders, and JSPI: Chrome and Edge 137 and later |

Every release has all three, because the JavaScript glue in yzma tests the browser and takes the one it can run. Each holds `yzma_wasm*.js` and `yzma_wasm*.wasm`.

These builds are not the same shape as the others. A WebAssembly module has no `dlopen`, so the backend cannot be a separate library, and TinyGo cannot compile the C++ of `llama.cpp` at all. So this repo also holds a small shim, `wasm/yzma_wasm.cpp`, which gives yzma an interface that it can call from a browser through JavaScript. The shim has its own version, and yzma refuses a module whose version it does not know.

The multimodal library of `llama.cpp`, mtmd, is in all three, so a model with a projector can look at an image. Emscripten 6.0.8 makes them, and the WebGPU build also needs the emdawnwebgpu package of Dawn.

See [wasm/README.md](./wasm/README.md) for how the shim works, how to build the modules, and what each variant costs in speed.

## How to check the latest version

```
VERSION=$(curl -s https://hybridgroup.github.io/llama-cpp-builder/version.json | jq -r '.tag_name')
```
