#!/bin/bash
# Usage: bash scripts/peek_result.sh <jsonl_path> <article_id>
# Pretty-prints the result line for <article_id> from a results JSONL file.

set -euo pipefail

JSONL="${1:?Usage: bash scripts/peek_result.sh <jsonl_path> <article_id>}"
ID="${2:?Usage: bash scripts/peek_result.sh <jsonl_path> <article_id>}"

LINE=$(grep -m1 -E "\"article_id\":\s*\"${ID}\"" "$JSONL" || true)
if [ -z "$LINE" ]; then
    echo "article_id ${ID} not found in ${JSONL}" >&2
    exit 1
fi
echo "$LINE" | python3 -m json.tool
