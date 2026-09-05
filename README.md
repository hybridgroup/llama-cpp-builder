# llama.cpp Builder

## Prebuilt binaries for the platforms that the normal builds leave out

This repo builds binary versions of `llama.cpp` libraries and executables for architectures that are not already part of the normal builds: Linux with CUDA or Vulkan, Linux arm64 with CPU, Vulkan, or OpenCL, and WebAssembly for a browser.

New releases are built automatically, and `llama.cpp` is checked twice per hour. A tagged
release such as `v0.4.0` is built first, and only until it is here, so a nightly build
published minutes later cannot take its place. Everything else builds the newest nightly.
The Build workflow also takes a `tag` to run by hand, which builds that tag whatever the
release pages say.

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

## Asset digests

Each release has a digest manifest that gives the SHA-256 of every asset that yzma can
install for that tag. A client can check an archive before it extracts it.

The manifest is an asset of the release it describes, and there is a copy on the site:

```
curl -sL https://github.com/hybridgroup/llama-cpp-builder/releases/download/b10783/b10783.json
curl -s https://hybridgroup.github.io/llama-cpp-builder/digests/b10783.json
```

Both hold the same bytes. The release asset is the one to prefer, because GitHub records
its digest. See [The manifest digest](#the-manifest-digest).

A tagged release such as `v0.3.0` has its own manifest, and a new nightly build does not
replace it.

yzma installs from two repositories, and an asset name can occur in both with different
bytes. So the manifest groups the assets by the repository that published them:

```json
{
  "version": 1,
  "tag": "v0.3.0",
  "upstream_tag": "b10621",
  "generated": "2026-09-03T14:03:39Z",
  "sources": {
    "hybridgroup/llama-cpp-builder": {
      "tag": "v0.3.0",
      "assets": {
        "llama-v0.3.0-bin-ubuntu-cuda-13-x64.tar.gz": {"sha256": "af61d03c..."}
      }
    },
    "ggml-org/llama.cpp": {
      "tag": "b10621",
      "assets": {
        "cudart-llama-bin-win-cuda-13.3-x64.zip": {"sha256": "1462a050..."}
      }
    }
  }
}
```

`tag` in a source block gives the release that holds those assets, so the download URL is
`https://github.com/<source>/releases/download/<source tag>/<asset name>`. For a tagged
release, the upstream assets are under the nightly build tag in `upstream_tag`, which
comes from the `nightly-tag.txt` asset.

### File digests

An asset that this repo builds also gives the digest of each file in it:

```json
"llama-b10783-bin-ubuntu-cpu-arm64.tar.gz": {
  "sha256": "5fcc5cbd...",
  "files": {"libllama.so.0.3.0": "9c2f...", "libggml.so.0.22.0": "1ab4..."},
  "links": {"libllama.so": "libllama.so.0", "libllama.so.0": "libllama.so.0.3.0"}
}
```

The names are the names that a client writes when it extracts the archive. A client
removes the archive after it extracts it, so these let it check an installation later.
`links` gives the name that each symbolic link points to, because a link has no bytes of
its own.

Each build job hashes its own output before it packs it, so the file digests cost no
download. Only this repo builds its assets, so an asset from `ggml-org/llama.cpp` has
`sha256` but no `files`. A client must accept an asset that has no `files`.

### How to check a download

Get the manifest for the tag, then compare the archive with the digest for its source:

```
TAG=b10783
ASSET=llama-$TAG-bin-ubuntu-cpu-arm64.tar.gz
curl -sO https://hybridgroup.github.io/llama-cpp-builder/digests/$TAG.json
curl -sLO https://github.com/hybridgroup/llama-cpp-builder/releases/download/$TAG/$ASSET

WANT=$(jq -r --arg a "$ASSET" \
  '.sources["hybridgroup/llama-cpp-builder"].assets[$a].sha256' $TAG.json)
echo "$WANT  $ASSET" | sha256sum -c
```

To check an installation after the archive is gone, compare the files in the library
directory:

```
jq -r --arg a "$ASSET" \
  '.sources["hybridgroup/llama-cpp-builder"].assets[$a].files
   | to_entries[] | "\(.value)  \(.key)"' $TAG.json > sums.txt
(cd /path/to/lib && sha256sum -c /path/to/sums.txt)
```

### The manifest digest

Every digest above lives in the manifest, so a client that takes the manifest on trust
takes all of them on trust. The manifest digest is the value that breaks that circle: a
client keeps it outside the release and checks the manifest bytes against it, then the
manifest checks everything else.

The manifest is published as an asset of its own release, named `<tag>.json`, and GitHub
records the SHA-256 of every asset it stores. So the manifest digest is published, with
the release and by the same means as the archives:

```
gh api repos/hybridgroup/llama-cpp-builder/releases/tags/b10816 \
  --jq '.assets[] | select(.name == "b10816.json") | .digest'
```

The release notes for the tag print the complete pin, and the version files carry it for
the two most recent builds, which costs no API request:

```console
$ curl -s https://hybridgroup.github.io/llama-cpp-builder/version.json
{"tag_name":"b10816","manifest_sha256":"<digest>","pin":"b10816@sha256:<digest>"}
```

yzma takes that pin as its version:

```
yzma install --version b10816@sha256:<digest> --lib /path/to/lib
```

The digest of a platform archive is not the manifest digest. Those digests are what the
manifest holds, one for each asset. The manifest never names itself, so `<tag>.json` is
left out of the asset list that it publishes.

### Where the manifests live

The manifests are in [digests/](./digests) in this repo. The release workflow copies them
to the site and uploads each one to its own release. A manifest is written one time and is
not rewritten, so a pin stays good. The manifests that seeded the directory hold only
`sha256`, because their releases are older than the build step that makes the file
digests.

The digests come from the GitHub release API, so they show that an archive is the archive
that was published. They are not a signature, and they do not show who built it. A pin
shows that the manifest is the one the client expected, which is a different thing.
