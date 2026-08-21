#!/bin/bash
# Usage: bash scripts/peek_result.sh <jsonl_path> <article_id>
# Pretty-prints EVERY result line for <article_id> from a results JSONL file.
# An article gets one record per asset group that fired, so a grader/mapper
# file holds several lines for the same article_id -- print them all, not just
# the first (the old `grep -m1` hid every tag but one).

set -euo pipefail

JSONL="${1:?Usage: bash scripts/peek_result.sh <jsonl_path> <article_id>}"
ID="${2:?Usage: bash scripts/peek_result.sh <jsonl_path> <article_id>}"

LINES=$(grep -E "\"article_id\":\s*\"${ID}\"" "$JSONL" || true)
if [ -z "$LINES" ]; then
    echo "article_id ${ID} not found in ${JSONL}" >&2
    exit 1
fi

N=$(printf '%s\n' "$LINES" | grep -c .)
echo "=== ${N} record(s) for article_id ${ID} in ${JSONL} ===" >&2
# One JSON object per line (JSONL) -- json.tool would choke on multiple, so
# pretty-print each line separately, blank line between.
printf '%s\n' "$LINES" | python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if line:
        print(json.dumps(json.loads(line), indent=4))
        print()
"
