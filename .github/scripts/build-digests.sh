#!/usr/bin/env bash
#
# Writes digests/<tag>.json, the SHA-256 digest of each release asset that a client
# can install for a llama.cpp tag.
#
# The assets come from two repositories. A name can occur in both with different
# bytes, so the manifest groups the assets by the repository that published them.
#
# When FILES_DIR names a directory that holds <asset>.files.json files, as
# hash-files.sh writes them, the digest of each file in those assets goes in too. A
# client can then check an installation after the archive is gone.
#
# Usage: build-digests.sh <tag> [output-dir]
# GH_TOKEN must hold a token that can read the two release pages.

set -euo pipefail

TAG="${1:?usage: build-digests.sh <tag> [output-dir]}"
OUT_DIR="${2:-digests}"

BUILDER_REPO="hybridgroup/llama-cpp-builder"
UPSTREAM_REPO="ggml-org/llama.cpp"

# A manifest that is published is never rewritten. A client can pin a tag and expect
# the digests to stay the same. Set FORCE=1 to correct a manifest by hand.
if [ -f "${OUT_DIR}/${TAG}.json" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "build-digests: ${OUT_DIR}/${TAG}.json exists, keep it"
  exit 0
fi

# A nightly tag such as "b10783" names its own upstream assets. A tagged release such
# as "v0.3.0" has no binaries of its own, so nightly-tag.txt gives the build that has
# them. This is the rule that yzma uses.
if [[ "$TAG" =~ ^b[0-9]+$ ]]; then
  UPSTREAM_TAG="$TAG"
else
  UPSTREAM_TAG=$(curl -sfL \
    "https://github.com/${UPSTREAM_REPO}/releases/download/${TAG}/nightly-tag.txt" |
    tr -d '[:space:]')
fi

if [[ ! "$UPSTREAM_TAG" =~ ^b[0-9]+$ ]]; then
  echo "build-digests: no upstream build tag for ${TAG}" >&2
  exit 1
fi

# assets_for prints {"<name>": {"sha256": "<hex>"}} for one release. GitHub gives the
# digest as "sha256:<hex>". An asset that is still uploading has no digest yet.
assets_for() {
  gh api "repos/$1/releases/tags/$2" --jq '
    [.assets[]
     | select(.digest != null)
     | {key: .name, value: {sha256: (.digest | sub("^sha256:"; ""))}}]
    | from_entries'
}

# missing_digests prints the names of the assets that have no digest.
missing_digests() {
  gh api "repos/$1/releases/tags/$2" --jq '.assets[] | select(.digest == null) | .name'
}

BUILDER_MISSING=$(missing_digests "$BUILDER_REPO" "$TAG")
if [ -n "$BUILDER_MISSING" ]; then
  echo "build-digests: ${BUILDER_REPO} ${TAG} has assets with no digest:" >&2
  echo "$BUILDER_MISSING" >&2
  exit 1
fi

BUILDER_ASSETS=$(assets_for "$BUILDER_REPO" "$TAG")
UPSTREAM_ASSETS=$(assets_for "$UPSTREAM_REPO" "$UPSTREAM_TAG")

# The build jobs hash their own output before they pack it, so the file digests cost
# no download. Only this repo builds its assets, so only its assets get them.
if [ -n "${FILES_DIR:-}" ] && [ -d "$FILES_DIR" ]; then
  WITH_FILES=0
  for name in $(jq -r 'keys[]' <<<"$BUILDER_ASSETS"); do
    [ -f "${FILES_DIR}/${name}.files.json" ] || continue
    BUILDER_ASSETS=$(jq \
      --arg name "$name" \
      --slurpfile contents "${FILES_DIR}/${name}.files.json" \
      '.[$name] += $contents[0]' <<<"$BUILDER_ASSETS")
    WITH_FILES=$((WITH_FILES + 1))
  done
  echo "build-digests: ${WITH_FILES} assets have file digests"
fi

if [ "$(jq 'length' <<<"$BUILDER_ASSETS")" -eq 0 ]; then
  echo "build-digests: ${BUILDER_REPO} ${TAG} published no assets" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# The keys stay sorted so that a second run of the same tag gives the same bytes.
jq -S -n \
  --arg tag "$TAG" \
  --arg upstream_tag "$UPSTREAM_TAG" \
  --arg builder_repo "$BUILDER_REPO" \
  --arg upstream_repo "$UPSTREAM_REPO" \
  --arg generated "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson builder_assets "$BUILDER_ASSETS" \
  --argjson upstream_assets "$UPSTREAM_ASSETS" \
  '{
    version: 1,
    tag: $tag,
    upstream_tag: $upstream_tag,
    generated: $generated,
    sources: {
      ($builder_repo): {tag: $tag, assets: $builder_assets},
      ($upstream_repo): {tag: $upstream_tag, assets: $upstream_assets}
    }
  }' > "${OUT_DIR}/${TAG}.json"

echo "build-digests: wrote ${OUT_DIR}/${TAG}.json"
echo "  ${BUILDER_REPO} ${TAG}: $(jq 'length' <<<"$BUILDER_ASSETS") assets"
echo "  ${UPSTREAM_REPO} ${UPSTREAM_TAG}: $(jq 'length' <<<"$UPSTREAM_ASSETS") assets"
