#!/bin/bash
# Usage: bash scripts/peek_article.sh <accession_number>
# Looks up a DJNW article by ID, pretty-prints it, and caches it locally.
# Optimized: extracts YYYY-MM from the accession number to target the right files.
# Also resolves tag codes to human-readable descriptions from the docs CSVs.

set -euo pipefail

# Resolve repo root from script location so this works regardless of which
# clone or which user runs it. Then cd so relative paths (article_cache,
# docs/) resolve to the repo, not the caller's cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ID="${1:?Usage: bash scripts/peek_article.sh <accession_number>}"
CACHE_DIR="article_cache"
# Lab-wide DJNW cleaned shards (override via env if your install differs).
DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles}"
DOCS_DIR="${DOCS_DIR:-$REPO_ROOT/docs}"
CACHE_FILE="${CACHE_DIR}/${ID}.json"

mkdir -p "$CACHE_DIR"

# Check cache first
if [ -f "$CACHE_FILE" ]; then
    echo "Cache hit: $CACHE_FILE" >&2
    cat "$CACHE_FILE"
else
    # Extract YYYY-MM from accession number (first 8 chars = YYYYMMDD)
    YEAR="${ID:0:4}"
    MONTH="${ID:4:2}"
    PATTERN="${YEAR}-${MONTH}"

    echo "Searching ${DATA_DIR}/${PATTERN}*_clean.jsonl for ${ID}..." >&2

    # Search only the targeted monthly files (typically 1-5 shards)
    FOUND=0
    for FILE in "${DATA_DIR}/${PATTERN}"*_clean.jsonl; do
        if [ ! -f "$FILE" ]; then
            continue
        fi
        MATCH=$(grep "\"${ID}\"" "$FILE" 2>/dev/null || true)
        if [ -n "$MATCH" ]; then
            echo "Found in: ${FILE}" >&2
            echo "$MATCH" | python3 -m json.tool > "$CACHE_FILE"
            echo "Cached to: ${CACHE_FILE}" >&2
            cat "$CACHE_FILE"
            FOUND=1
            break
        fi
    done

    if [ "$FOUND" -eq 0 ]; then
        echo "Article ${ID} not found in ${PATTERN}* files" >&2
        exit 1
    fi
fi

# Resolve tag codes to descriptions from docs CSVs
echo "" >&2
echo "=== Tag Descriptions ===" >&2

python3 - "$CACHE_FILE" "$DOCS_DIR" <<'PYEOF'
import csv, json, sys
from pathlib import Path

cache_file = sys.argv[1]
docs_dir = Path(sys.argv[2])

# Build lookup from all docs CSVs
lookup = {}
for csvf in sorted(docs_dir.glob("*.csv")):
    with open(csvf) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                code = row[0].strip()
                desc = row[1].strip()
                if code and code != "symbol" and not code.startswith("#"):
                    lookup[code] = desc

# Read the cached article
with open(cache_file) as f:
    article = json.load(f)

codes_obj = article.get("codes", {})
fields = ["subject", "industry", "market", "product", "geo"]

for field in fields:
    vals = codes_obj.get(field, [])
    if not vals:
        continue
    items = vals if isinstance(vals, list) else [vals]
    if not items:
        continue
    print(f"\n  {field}:", file=sys.stderr)
    for code in items:
        if isinstance(code, dict):
            code = code.get("code") or code.get("isin", "")
        code = str(code)
        desc = lookup.get(code, "??? (undocumented)")
        print(f"    {code:<16s} {desc}", file=sys.stderr)

# Also show company codes if present
for field in ["company", "isin"]:
    vals = codes_obj.get(field, [])
    if not vals:
        continue
    items = vals if isinstance(vals, list) else [vals]
    if not items:
        continue
    print(f"\n  {field}:", file=sys.stderr)
    for item in items:
        if isinstance(item, dict):
            code = item.get("code") or item.get("isin", "")
        else:
            code = str(item)
        print(f"    {code}", file=sys.stderr)
PYEOF

# Show the article body split by the same paragraph splitter the pipeline
# feeds the LLM (src.loaders.split_into_paragraphs), with [i] indices.
echo "" >&2
echo "=== Paragraphs (as fed to LLM) ===" >&2

python3 - "$CACHE_FILE" "$REPO_ROOT" <<'PYEOF'
import json, sys
from pathlib import Path

cache_file = sys.argv[1]
repo_root = Path(sys.argv[2])
sys.path.insert(0, str(repo_root))

from src.loaders import split_into_paragraphs

with open(cache_file) as f:
    article = json.load(f)

text = article.get("text", "")
paragraphs = split_into_paragraphs(text)

for i, p in enumerate(paragraphs):
    print(f"[{i}] {p}\n", file=sys.stderr)
PYEOF
