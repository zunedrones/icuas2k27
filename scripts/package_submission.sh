#!/usr/bin/env bash
# Host: build the Phase 1 submission zip (rulebook section 6) from the committed sources.
#   ./scripts/package_submission.sh [output.zip]
set -e
cd "$(dirname "$0")/.."
OUT="${1:-icuas2k27_submission_$(git rev-parse --short HEAD).zip}"
if [ -n "$(git status --porcelain)" ]; then
  echo "warning: uncommitted changes are NOT included (git archive packs HEAD)" >&2
fi
git archive --format=zip --prefix=icuas2k27/ -o "$OUT" HEAD
echo "$OUT ($(du -h "$OUT" | cut -f1))"
