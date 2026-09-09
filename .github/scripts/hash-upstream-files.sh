#!/usr/bin/env bash
#
# Writes <asset>.files.json for an upstream release asset that this repo does not build.
#
# The build jobs hash the files that they pack, so a builder asset costs no download. An
# upstream asset must be downloaded and unpacked to get the same digests. Only the macOS
# arm64 archive gets them, because that archive is what a macOS arm64 client installs.
#
# Usage: hash-upstream-files.sh <tag> [output-dir]
# GH_TOKEN must hold a token that can read the upstream release page.

set -euo pipefail

TAG="${1:?usage: hash-upstream-files.sh <tag> [output-dir]}"
OUT_DIR="${2:-upstream-files-digests}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/upstream-tag.sh"

UPSTREAM_TAG=$(upstream_tag_for "$TAG")

# The tag in the name is the upstream build tag, not the tag of this repo.
export ASSET="llama-${UPSTREAM_TAG}-bin-macos-arm64.tar.gz"

WANT=$(gh api "repos/${UPSTREAM_REPO}/releases/tags/${UPSTREAM_TAG}" \
  --jq '.assets[] | select(.name == env.ASSET) | .digest // ""' | sed 's/^sha256://')

if [ -z "$WANT" ]; then
  echo "hash-upstream-files: ${UPSTREAM_REPO} ${UPSTREAM_TAG} has no digest for ${ASSET}" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

curl -sfL -o "${WORK}/${ASSET}" \
  "https://github.com/${UPSTREAM_REPO}/releases/download/${UPSTREAM_TAG}/${ASSET}"

GOT=$(sha256sum "${WORK}/${ASSET}" | cut -d' ' -f1)
if [ "$GOT" != "$WANT" ]; then
  echo "hash-upstream-files: ${ASSET} has the digest '${GOT}', want '${WANT}'" >&2
  exit 1
fi

# The archive holds one top directory, and a client removes the first path element when it
# extracts. So the names here are the names that a client writes.
mkdir -p "${WORK}/extract"
tar -xzf "${WORK}/${ASSET}" --strip-components=1 --no-same-owner -C "${WORK}/extract"

cd "$OUT_DIR"
"${SCRIPT_DIR}/hash-files.sh" "$ASSET" "${WORK}/extract"

echo "hash-upstream-files: wrote ${OUT_DIR}/${ASSET}.files.json"
