# CMake project for the yzma WebAssembly build of llama.cpp.
#
# Configure it with emcmake so that Emscripten sets the toolchain:
#
#   emcmake cmake -S wasm -B build-wasm -DLLAMA_CPP_DIR=../llama.cpp
#
# Use wasm/build.sh instead of calling cmake directly. The script sets all of
# the flags that the browser needs.

cmake_minimum_required(VERSION 3.20)
project(yzma_wasm CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "This project needs Emscripten. Configure it with emcmake cmake.")
endif()

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH
    "Path to the llama.cpp sources")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "No llama.cpp sources at ${LLAMA_CPP_DIR}")
endif()

# A WebAssembly module has no dlopen, so the backend cannot be a separate
# shared library. This is the opposite of the native builds, which use
# GGML_BACKEND_DL=ON and GGML_CPU_ALL_VARIANTS=ON.
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL        OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS  OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE            OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP            OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP        OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL             OFF CACHE BOOL "" FORCE)

# mtmd is the multimodal library of llama.cpp, and it lives under tools/, which
# llama.cpp skips for an Emscripten build. LLAMA_BUILD_MTMD is the hook upstream
# has for taking the library on its own, and it fires because LLAMA_BUILD_TOOLS
# and LLAMA_BUILD_COMMON are both off, so this builds the library and none of the
# programs of tools/.
#
# mtmd links only ggml and llama in public, so it needs nothing from the common
# library of llama.cpp. Video needs a subprocess for ffmpeg, and llama.cpp turns
# subprocesses off for Emscripten, which turns video off with them.
set(LLAMA_BUILD_MTMD       ON  CACHE BOOL "" FORCE)

# llama.cpp defaults to 64-bit memory for an Emscripten build. This build stays
# with 32-bit memory, because a 64-bit build makes every pointer an i64, which
# reaches JavaScript as a BigInt, and the Go code in pkg/llamawasm works with
# numbers. The cost is the limit of 4 GB of memory in the module.
set(LLAMA_WASM_MEM64       OFF CACHE BOOL "" FORCE)

# The HTML output of llama.cpp changes the suffix of every executable, and the
# single file output of ggml puts the WebAssembly inside the JavaScript.
set(LLAMA_BUILD_HTML       OFF CACHE BOOL "" FORCE)
set(LLAMA_WASM_SINGLE_FILE OFF CACHE BOOL "" FORCE)
set(GGML_WASM_SINGLE_FILE  OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" llama.cpp EXCLUDE_FROM_ALL)

add_executable(yzma_wasm yzma_wasm.cpp)
target_link_libraries(yzma_wasm PRIVATE llama ggml mtmd)

# YZMA_WASM_EXPORTS and YZMA_WASM_LINK_FLAGS come from build.sh.
if(DEFINED YZMA_WASM_EXPORTS)
    target_link_options(yzma_wasm PRIVATE "-sEXPORTED_FUNCTIONS=${YZMA_WASM_EXPORTS}")
endif()

set_target_properties(yzma_wasm PROPERTIES OUTPUT_NAME "${YZMA_WASM_OUTPUT_NAME}" SUFFIX ".js")
