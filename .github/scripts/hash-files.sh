#!/usr/bin/env bash
#
# Writes <archive>.files.json, the SHA-256 of each file that goes into an archive.
#
# Run it on the directory that the archive is packed from, before the pack step. The
# names are relative to that directory, which is what a client sees after it extracts
# the archive and removes the top directory.
#
# The build runners do not all have jq, so this writes the JSON itself.
#
# Usage: hash-files.sh <archive-name> <source-dir>

set -euo pipefail

ARCHIVE="${1:?usage: hash-files.sh <archive-name> <source-dir>}"
SRC_DIR="${2:?usage: hash-files.sh <archive-name> <source-dir>}"

OUT="$(pwd)/${ARCHIVE}.files.json"

# emit_map reads "value<TAB>key" lines and writes them as a JSON object. A backslash
# or a quotation mark in a name is escaped.
emit_map() {
  local first=1 key value
  printf '{'
  while IFS="$(printf '\t')" read -r value key; do
    [ -n "$key" ] || continue
    if [ "$first" -eq 1 ]; then
      first=0
    else
      printf ','
    fi
    key=$(printf '%s' "$key" | sed 's/\\/\\\\/g; s/"/\\"/g')
    value=$(printf '%s' "$value" | sed 's/\\/\\\\/g; s/"/\\"/g')
    printf '\n    "%s": "%s"' "$key" "$value"
  done
  [ "$first" -eq 1 ] || printf '\n  '
  printf '}'
}

cd "$SRC_DIR"

# A regular file gives its digest. A symbolic link has no bytes of its own, so it
# gives the name it points to. An archive holds both, and a client writes both.
FILE_LINES=$(find . -type f -print0 | sort -z | xargs -0 -r sha256sum |
  sed "s|^\([0-9a-f]\{64\}\)  \./|\1$(printf '\t')|")
LINK_LINES=$(find . -type l -printf '%l\t%p\n' | sort | sed "s|$(printf '\t')\./|$(printf '\t')|")

{
  printf '{\n  "files": '
  printf '%s\n' "$FILE_LINES" | emit_map
  printf ',\n  "links": '
  printf '%s\n' "$LINK_LINES" | emit_map
  printf '\n}\n'
} > "$OUT"

count() { [ -z "$1" ] && echo 0 || printf '%s\n' "$1" | wc -l; }
echo "hash-files: ${ARCHIVE} has $(count "$FILE_LINES") files and $(count "$LINK_LINES") links"
