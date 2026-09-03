#!/usr/bin/env bash
#
# Writes <archive>.files.json, the SHA-256 of each file that goes into an archive.
#
# Run it on the directory that the archive is packed from, before the pack step. The
# names are relative to that directory, which is what a client sees after it extracts
# the archive and removes the top directory.
#
# Usage: hash-files.sh <archive-name> <source-dir>

set -euo pipefail

ARCHIVE="${1:?usage: hash-files.sh <archive-name> <source-dir>}"
SRC_DIR="${2:?usage: hash-files.sh <archive-name> <source-dir>}"

OUT="$(pwd)/${ARCHIVE}.files.json"

cd "$SRC_DIR"

# A regular file gives its digest. A symbolic link has no bytes of its own, so it
# gives the name it points to. An archive holds both, and a client writes both.
FILES=$(find . -type f -print0 | sort -z | xargs -0 -r sha256sum |
  jq -R -s '
    split("\n")
    | map(select(length > 0))
    | map(capture("^(?<sha>[0-9a-f]{64})\\s+\\./(?<name>.*)$"))
    | map({key: .name, value: .sha})
    | from_entries')

LINKS=$(find . -type l -printf '%p\t%l\n' | sort |
  jq -R -s '
    split("\n")
    | map(select(length > 0))
    | map(split("\t") | {key: (.[0] | ltrimstr("./")), value: .[1]})
    | from_entries')

jq -S -n --argjson files "$FILES" --argjson links "$LINKS" \
  '{files: $files, links: $links}' > "$OUT"

echo "hash-files: ${ARCHIVE} has $(jq 'length' <<<"$FILES") files and $(jq 'length' <<<"$LINKS") links"
