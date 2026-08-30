#!/usr/bin/env bash
#
# build.sh makes the yzma WebAssembly build of llama.cpp.
#
# Usage:
#   wasm/build.sh [--mt] [--llama-cpp DIR] [--build-dir DIR] [--out DIR]
#
#   --mt          Make the multiple thread build. The default is the single
#                 thread build.
#   --llama-cpp   Path to the llama.cpp sources. The default is ./llama.cpp.
#   --build-dir   Directory for the intermediate files.
#   --out         Directory for yzma_wasm.js and yzma_wasm.wasm.
#
# The Emscripten SDK must be active. Run "source /path/to/emsdk_env.sh" first.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"

mt=0
llama_cpp="$repo/llama.cpp"
build_dir=""
out_dir="$repo/build-wasm/out"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mt)        mt=1; shift ;;
        --llama-cpp) llama_cpp="$2"; shift 2 ;;
        --build-dir) build_dir="$2"; shift 2 ;;
        --out)       out_dir="$2"; shift 2 ;;
        -h|--help)   sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if ! command -v emcmake >/dev/null 2>&1; then
    echo "emcmake is not in PATH. Activate the Emscripten SDK first." >&2
    exit 1
fi

if [[ $mt -eq 1 ]]; then
    name="yzma_wasm_mt"
else
    name="yzma_wasm"
fi
if [[ -z "$build_dir" ]]; then
    build_dir="$repo/build-wasm/$name"
fi

# The list of exported functions comes from the shim, so the two cannot get out
# of step. Emscripten needs a leading underscore on each name.
exports="$(grep -oE '^(int|void|float) (yzma_[a-z0-9_]+)\(' "$here/yzma_wasm.cpp" \
    | sed -E 's/^(int|void|float) /_/; s/\($//' | sort -u | paste -sd, -)"
exports="_malloc,_free,${exports}"
echo "exported functions: $exports"

# Flags for the browser:
#   MODULARIZE + EXPORT_NAME       the JavaScript glue makes the instance
#   ALLOW_MEMORY_GROWTH            a model needs much more than the start size
#   MAXIMUM_MEMORY=4GB             the largest that wasm32 can address
#   FORCE_FILESYSTEM               the Go code puts the model in MEMFS
#   EXPORTED_RUNTIME_METHODS       what pkg/llamawasm uses from JavaScript
link_flags=(
    "-O3"
    "-msimd128"
    "-fwasm-exceptions"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=yzmaModule"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sMAXIMUM_MEMORY=4294967296"
    "-sINITIAL_MEMORY=134217728"
    "-sSTACK_SIZE=1048576"
    "-sFORCE_FILESYSTEM=1"
    "-sEXIT_RUNTIME=0"
    "-sASSERTIONS=0"
    "-sEXPORTED_RUNTIME_METHODS=FS,HEAPU8,HEAP32,HEAPF32,stringToUTF8,lengthBytesUTF8,UTF8ToString"
)

# llama.cpp throws and catches exceptions while it loads a model, so the build
# needs real exception support. -fwasm-exceptions uses the exception handling
# of WebAssembly itself, which all current browsers have.
compile_flags=("-O3" "-msimd128" "-fwasm-exceptions")

if [[ $mt -eq 1 ]]; then
    # A multiple thread build needs SharedArrayBuffer, which a browser gives
    # only to a page that sets the COOP and COEP headers. The JavaScript glue
    # tests for this and takes the single thread build if it is not available.
    link_flags+=("-pthread" "-sSHARED_MEMORY=1" "-sPTHREAD_POOL_SIZE=8" "-sPROXY_TO_PTHREAD=0")
    compile_flags+=("-pthread")
fi

echo "configuring $name in $build_dir"
emcmake cmake -S "$here" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CPP_DIR="$llama_cpp" \
    -DYZMA_WASM_OUTPUT_NAME="$name" \
    -DYZMA_WASM_EXPORTS="$exports" \
    -DCMAKE_CXX_FLAGS="${compile_flags[*]}" \
    -DCMAKE_C_FLAGS="${compile_flags[*]}" \
    -DCMAKE_EXE_LINKER_FLAGS="${link_flags[*]}"

echo "building $name"
cmake --build "$build_dir" --config Release -j "$(nproc 2>/dev/null || echo 4)"

mkdir -p "$out_dir"
cp "$build_dir/$name.js" "$build_dir/$name.wasm" "$out_dir/"
# A multiple thread build of an older Emscripten also makes a worker file.
if [[ -f "$build_dir/$name.worker.js" ]]; then
    cp "$build_dir/$name.worker.js" "$out_dir/"
fi

echo "output in $out_dir:"
ls -la "$out_dir"
