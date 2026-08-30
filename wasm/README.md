# WebAssembly build

This directory makes the WebAssembly build of llama.cpp that
[yzma](https://github.com/hybridgroup/yzma) uses in a browser, through the
`pkg/llamawasm` package.

## Why there is a shim

On a native platform yzma calls llama.cpp with libffi and loads the shared
libraries at run time. WebAssembly has no `dlopen` and no libffi, and TinyGo
cannot compile C++ for WebAssembly. The Go code therefore calls llama.cpp
through JavaScript, and `yzma_wasm.cpp` gives it an interface that is possible
to call that way:

- Scalar arguments only. No struct crosses the boundary, so the Go code does
  not need the wasm32 layout of any llama.cpp struct.
- Small `int32` handles for every object. The Go code never holds a
  WebAssembly address.
- A negative return value is an error. `yzma_last_error` gives the text.

The generation loop stays in Go: one `yzma_decode` and one
`yzma_sampler_sample` for each token.

## ABI

`yzma_abi_version` in `yzma_wasm.cpp` is the version of this interface.

| Version | What it adds |
| --- | --- |
| 1 | The calls for text generation and embeddings. |
| 2 | `yzma_gpu_device`, which names the device that is not the CPU. |

`pkg/llamawasm` drives every version from 1 up to the one it knows, and tests
for a call before it uses it, so a new yzma still works with the modules of an
older release.

**Increase it each time you add, remove, or change a function.** The
`pkg/llamawasm` package compares the value when it starts and refuses a module
that does not match, so a version that is out of step becomes a clear error
instead of a wrong result.

The list of exported functions is not written twice: `build.sh` reads it out of
`yzma_wasm.cpp`.

## Build

Activate the Emscripten SDK, put the llama.cpp sources in `llama.cpp/`, then:

```
wasm/build.sh              # single thread, CPU
wasm/build.sh --mt         # multiple threads, CPU
wasm/build.sh --webgpu     # GPU
```

The output is in `build-wasm/out`.

## The WebGPU build

`--webgpu` puts the computation on the GPU of the browser, with
`-DGGML_WEBGPU=ON`. Three things come with it:

- **emdawnwebgpu.** Emscripten has no WebGPU bindings of its own, so the backend
  uses the package that Dawn makes. The backend follows one commit of Dawn, so
  `DAWN_TAG` in `build.sh` pins the package, and the script downloads it if
  `--emdawn` gives no path. This is the same tag that the llama.cpp CI uses.
- **JSPI.** Asking for a GPU adapter is asynchronous, and the backend waits for
  it inside a synchronous C++ call, so the build needs JavaScript Promise
  Integration. `GGML_WEBGPU_JSPI` is on by default and keeps
  `-fwasm-exceptions`, the same as the builds on the CPU. JSPI needs Chrome or
  Edge 137 and later; the glue in yzma tests for it and takes a build on the CPU
  in a browser that has none.
- **`JSPI_EXPORTS`.** Emscripten makes only `main` able to suspend. `build.sh`
  therefore names the calls of the shim that reach the GPU, and each of them
  gives a promise to JavaScript. The calls that only work on strings and tokens
  stay synchronous, because the generation loop makes several of them for each
  token.

One thread only: the GPU does the work, and threads with JSPI are a fragile
mix.

A GPU is not enough on its own. The backend needs an adapter that supports f16
shaders, and it reports no device without one, so `yzma_gpu_device` says whether
llama.cpp really has a GPU.

## The three variants

The multiple thread build needs `SharedArrayBuffer`, which a browser gives only
to a page that sets these headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The JavaScript glue in yzma tests for WebGPU and for this, and takes the build
that the browser can run, so all three variants ship in every release:

- `llama-<tag>-bin-wasm-simd.tar.gz`
- `llama-<tag>-bin-wasm-simd-mt.tar.gz`
- `llama-<tag>-bin-wasm-webgpu.tar.gz`

## Limits

- The builds on the CPU use SIMD. The WebGPU build needs Chrome or Edge 137 and
  later, and an adapter with f16 shaders.
- A browser does not get the matrix instructions of a subgroup, which are behind
  a toggle that only Dawn outside a browser has, so the GPU is slower in a page
  than the same backend is on a desktop.
- An operation larger than `maxStorageBufferBindingSize` goes back to the CPU.
- wasm32 can address 4 GB, and one `ArrayBuffer` holds at most 2 GB, so a large
  model needs `llama-gguf-split`.
