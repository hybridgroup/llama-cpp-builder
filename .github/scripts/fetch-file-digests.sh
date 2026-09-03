#!/usr/bin/env bash
#
# Downloads the per-file digest artifacts that the build jobs of a run made.
#
# Usage: fetch-file-digests.sh <run-id> [output-dir]
# GH_TOKEN must hold a token that can read the artifacts of the run.

set -euo pipefail

RUN_ID="${1:?usage: fetch-file-digests.sh <run-id> [output-dir]}"
OUT_DIR="${2:-files-digests}"
REPO="${GITHUB_REPOSITORY:-hybridgroup/llama-cpp-builder}"

mkdir -p "$OUT_DIR"

COUNT=0
while IFS=$'\t' read -r name url; do
  [ -n "$name" ] || continue
  curl -sfL -H "Accept: application/vnd.github+json" \
    -H "Authorization: token ${GH_TOKEN}" \
    -o "${OUT_DIR}/${name}" "$url"
  COUNT=$((COUNT + 1))
done < <(
  gh api --paginate \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/${REPO}/actions/runs/${RUN_ID}/artifacts" \
    --jq '.artifacts[] | select(.name | endswith(".files.json")) | "\(.name)\t\(.archive_download_url)"'
)

echo "fetch-file-digests: got ${COUNT} file digest sets into ${OUT_DIR}"
