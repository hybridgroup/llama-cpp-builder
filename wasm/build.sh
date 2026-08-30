#!/usr/bin/env bash
#
# build.sh makes the yzma WebAssembly build of llama.cpp.
#
# Usage:
#   wasm/build.sh [--mt|--webgpu] [--llama-cpp DIR] [--build-dir DIR] [--out DIR]
#                 [--emdawn DIR]
#
#   --mt          Make the multiple thread build. The default is the single
#                 thread build on the CPU.
#   --webgpu      Make the WebGPU build. It uses one thread, because the GPU
#                 does the work.
#   --llama-cpp   Path to the llama.cpp sources. The default is ./llama.cpp.
#   --build-dir   Directory for the intermediate files.
#   --out         Directory for the .js and .wasm files.
#   --emdawn      Path to an emdawnwebgpu_pkg directory for --webgpu. The
#                 script downloads the pinned package if this is not given.
#
# The Emscripten SDK must be active. Run "source /path/to/emsdk_env.sh" first.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"

# DAWN_TAG is the emdawnwebgpu package that goes with the WebGPU backend of
# llama.cpp. The backend follows one commit of Dawn, so the package and the
# backend must stay in step. This is the same tag that the llama.cpp CI uses.
DAWN_TAG="${DAWN_TAG:-v20260317.182325}"

mt=0
webgpu=0
llama_cpp="$repo/llama.cpp"
build_dir=""
out_dir="$repo/build-wasm/out"
emdawn_dir="${EMDAWNWEBGPU_DIR:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mt)        mt=1; shift ;;
        --webgpu)    webgpu=1; shift ;;
        --llama-cpp) llama_cpp="$2"; shift 2 ;;
        --build-dir) build_dir="$2"; shift 2 ;;
        --out)       out_dir="$2"; shift 2 ;;
        --emdawn)    emdawn_dir="$2"; shift 2 ;;
        -h|--help)   sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ $mt -eq 1 && $webgpu -eq 1 ]]; then
    echo "--mt and --webgpu together are not supported: the WebGPU build uses one thread." >&2
    exit 2
fi

if ! command -v emcmake >/dev/null 2>&1; then
    echo "emcmake is not in PATH. Activate the Emscripten SDK first." >&2
    exit 1
fi

case "$mt$webgpu" in
    10) name="yzma_wasm_mt" ;;
    01) name="yzma_wasm_webgpu" ;;
    *)  name="yzma_wasm" ;;
esac
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
    "-sEXPORTED_RUNTIME_METHODS=FS,HEAPU8,HEAP32,HEAPF32,ccall,stringToUTF8,lengthBytesUTF8,UTF8ToString"
)

# llama.cpp throws and catches exceptions while it loads a model, so the build
# needs real exception support. -fwasm-exceptions uses the exception handling
# of WebAssembly itself, which all current browsers have.
compile_flags=("-O3" "-msimd128" "-fwasm-exceptions")

cmake_args=()

if [[ $mt -eq 1 ]]; then
    # A multiple thread build needs SharedArrayBuffer, which a browser gives
    # only to a page that sets the COOP and COEP headers. The JavaScript glue
    # tests for this and takes the single thread build if it is not available.
    link_flags+=("-pthread" "-sSHARED_MEMORY=1" "-sPTHREAD_POOL_SIZE=8" "-sPROXY_TO_PTHREAD=0")
    compile_flags+=("-pthread")
fi

if [[ $webgpu -eq 1 ]]; then
    # Emscripten has no WebGPU bindings of its own, so the backend uses the
    # emdawnwebgpu package of Dawn. Take the pinned package if the caller gives
    # no path, because the package must match the Dawn commit that the backend
    # follows.
    if [[ -z "$emdawn_dir" ]]; then
        emdawn_dir="$repo/build-wasm/emdawnwebgpu_pkg"
        if [[ ! -f "$emdawn_dir/emdawnwebgpu.port.py" ]]; then
            echo "downloading emdawnwebgpu ${DAWN_TAG}"
            mkdir -p "$repo/build-wasm"
            curl -fsSL -o "$repo/build-wasm/emdawn.zip" \
                "https://github.com/google/dawn/releases/download/${DAWN_TAG}/emdawnwebgpu_pkg-${DAWN_TAG}.zip"
            unzip -q -o "$repo/build-wasm/emdawn.zip" -d "$repo/build-wasm"
        fi
    fi

    if [[ ! -f "$emdawn_dir/emdawnwebgpu.port.py" ]]; then
        echo "no emdawnwebgpu package at $emdawn_dir" >&2
        exit 1
    fi
    echo "emdawnwebgpu: $emdawn_dir"

    # GGML_WEBGPU_JSPI is on by default, which is the path that the llama.cpp CI
    # builds. It keeps -fwasm-exceptions, the same as the builds on the CPU. The
    # -sJSPI link flag arrives on its own from the ggml-webgpu target.
    #
    # JSPI needs a browser that has WebAssembly JavaScript Promise Integration,
    # which is Chrome and Edge 137 and later. The JavaScript glue tests for
    # WebGPU before it takes this build, and takes a build on the CPU if the
    # browser cannot use it.
    cmake_args+=(
        "-DGGML_WEBGPU=ON"
        "-DEMDAWNWEBGPU_DIR=$emdawn_dir"
        "-DLLAMA_OPENSSL=OFF"
    )

    # JSPI wraps only main by default, and a call that suspends without a
    # wrapper stops the program. These are the calls that reach the GPU, so
    # these are the ones that must be able to suspend. Each of them gives a
    # promise to JavaScript, and pkg/llamawasm waits for it.
    #
    # The calls that are not here stay synchronous: they only work on strings
    # and tokens in the memory of the module, and the generation loop makes
    # several of them for each token.
    # JSPI_EXPORTS takes the names of the exports of the WebAssembly module,
    # which have no leading underscore. EXPORTED_FUNCTIONS takes the same names
    # with one. A name with an underscore here matches nothing, and then the
    # first call that tries to suspend stops the program with "trying to suspend
    # without WebAssembly.promising".
    jspi_exports=(
        "yzma_backend_init"
        "yzma_backend_free"
        "yzma_gpu_device"
        "yzma_model_load"
        "yzma_model_free"
        "yzma_context_new"
        "yzma_context_free"
        "yzma_memory_clear"
        "yzma_decode"
        "yzma_encode"
        "yzma_get_embeddings_seq"
        "yzma_sampler_sample"
        "yzma_mtmd_init_from_file"
        "yzma_mtmd_free"
        "yzma_mtmd_tokenize"
        "yzma_mtmd_helper_eval_chunks"
    )

    # Each one must also be in the list of exported functions, which has the
    # underscore.
    for jspi_name in "${jspi_exports[@]}"; do
        if [[ ",${exports}," != *",_${jspi_name},"* ]]; then
            echo "$jspi_name is in JSPI_EXPORTS but the shim does not export it" >&2
            exit 1
        fi
    done

    link_flags+=("-sJSPI_EXPORTS=$(IFS=,; echo "${jspi_exports[*]}")")
fi

echo "configuring $name in $build_dir"
emcmake cmake -S "$here" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CPP_DIR="$llama_cpp" \
    -DYZMA_WASM_OUTPUT_NAME="$name" \
    -DYZMA_WASM_EXPORTS="$exports" \
    -DCMAKE_CXX_FLAGS="${compile_flags[*]}" \
    -DCMAKE_C_FLAGS="${compile_flags[*]}" \
    -DCMAKE_EXE_LINKER_FLAGS="${link_flags[*]}" \
    "${cmake_args[@]+"${cmake_args[@]}"}"

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
