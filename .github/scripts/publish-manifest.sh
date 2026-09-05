#!/usr/bin/env bash
#
# Publishes digests/<tag>.json as an asset of the release that it describes, so that
# GitHub records the SHA-256 of the manifest. That value is what a client pins with
# "<tag>@sha256:<digest>", and it is the only digest here that does not come from the
# same file it protects.
#
# The pin also goes in the release notes, so a person can read it without the API.
#
# The build workflow runs this for a new release. It also back-fills a release that was
# made before the manifest was an asset, one or more tags at a time.
#
# Usage: publish-manifest.sh <tag>...
# GH_TOKEN must hold a token that can write to the releases of this repo.

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: publish-manifest.sh <tag>..." >&2
  exit 1
fi

REPO="hybridgroup/llama-cpp-builder"

# publish_one uploads the manifest for one tag and prints the pin.
publish_one() {
  local tag="$1"
  local manifest="digests/${tag}.json"

  if [ ! -f "$manifest" ]; then
    echo "publish-manifest: ${manifest} is not there" >&2
    return 1
  fi

  local want
  want=$(sha256sum "$manifest" | cut -d' ' -f1)

  # --clobber lets a repeat run replace an asset that is already there. The bytes are
  # the same, because build-digests.sh never rewrites a manifest it has published.
  gh release upload "$tag" "$manifest" --clobber --repo "$REPO"

  # GitHub computes the digest after the upload, so it reads as null for a moment.
  export MANIFEST_ASSET="${tag}.json"
  local got=""
  for _ in 1 2 3 4 5 6; do
    got=$(gh api "repos/${REPO}/releases/tags/${tag}" \
      --jq '.assets[] | select(.name == env.MANIFEST_ASSET) | .digest // ""')
    if [ -n "$got" ]; then
      break
    fi
    sleep 5
  done

  if [ "$got" != "sha256:${want}" ]; then
    echo "publish-manifest: ${tag} has the digest '${got}', want 'sha256:${want}'" >&2
    return 1
  fi

  local pin="${tag}@sha256:${want}"
  echo "publish-manifest: ${pin}"

  # The notes are written once. A repeat run must not add the pin a second time.
  local body
  body=$(gh release view "$tag" --repo "$REPO" --json body --jq '.body')
  if [[ "$body" == *"$pin"* ]]; then
    return 0
  fi

  gh release edit "$tag" --repo "$REPO" --notes "${body}

## Pinned version

The digest below is the SHA-256 of \`${tag}.json\`, the manifest that names the digest of
every other asset of this release. It is not the digest of a platform archive.

\`\`\`
${pin}
\`\`\`
"
}

for tag in "$@"; do
  publish_one "$tag"
done
