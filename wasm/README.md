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
**Increase it each time you add, remove, or change a function.** The
`pkg/llamawasm` package compares the value when it starts and refuses a module
that does not match, so a version that is out of step becomes a clear error
instead of a wrong result.

The list of exported functions is not written twice: `build.sh` reads it out of
`yzma_wasm.cpp`.

## Build

Activate the Emscripten SDK, put the llama.cpp sources in `llama.cpp/`, then:

```
wasm/build.sh              # single thread
wasm/build.sh --mt         # multiple threads
```

The output is `build-wasm/out/yzma_wasm.js` and `yzma_wasm.wasm`.

## The two variants

The multiple thread build needs `SharedArrayBuffer`, which a browser gives only
to a page that sets these headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The JavaScript glue in yzma tests for this and takes the single thread build if
the page is not isolated, so both variants ship in every release:

- `llama-<tag>-bin-wasm-simd.tar.gz`
- `llama-<tag>-bin-wasm-simd-mt.tar.gz`

## Limits

- CPU and SIMD only. There is no GPU backend.
- wasm32 can address 4 GB, and one `ArrayBuffer` holds at most 2 GB, so a large
  model needs `llama-gguf-split`.
