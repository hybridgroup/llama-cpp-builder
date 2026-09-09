#!/usr/bin/env bash
#
# upstream_tag_for prints the llama.cpp build tag that holds the upstream assets for a tag
# of this repo.
#
# A nightly tag such as "b10783" names its own upstream assets. A tagged release such as
# "v0.3.0" has no binaries of its own, so nightly-tag.txt gives the build that has them.
# This is the rule that yzma uses.
#
# Source this file. It defines a function and does nothing else.

UPSTREAM_REPO="ggml-org/llama.cpp"

upstream_tag_for() {
  local tag="${1:?usage: upstream_tag_for <tag>}"
  local upstream_tag

  if [[ "$tag" =~ ^b[0-9]+$ ]]; then
    upstream_tag="$tag"
  else
    upstream_tag=$(curl -sfL \
      "https://github.com/${UPSTREAM_REPO}/releases/download/${tag}/nightly-tag.txt" |
      tr -d '[:space:]')
  fi

  if [[ ! "$upstream_tag" =~ ^b[0-9]+$ ]]; then
    echo "upstream-tag: no upstream build tag for ${tag}" >&2
    return 1
  fi

  printf '%s\n' "$upstream_tag"
}
