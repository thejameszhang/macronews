"""Runner: streams a raw .nml file → writes one TabularResult per article
to a sidecar JSONL.

CLI mirrors src/grading/runner.py:

  python src/tabular/runner.py \\
      --nml-file /nfs/roberts/project/pi_btk22/rc2573/DJN/2002-09.nml \\
      --output results/tabular/2002-09.jsonl

Each input line in the sidecar is a TabularResult.model_dump_json() row
keyed by accession_number. Multi-doc continuations (same accession across
"-1-", "-2-", "-3-" headlines) are aggregated upstream by the NML reader.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tabular.detector import classify_pre_block  # noqa: E402
from tabular.nml_reader import NMLArticle, count_p_body_tokens, iter_nml_articles  # noqa: E402
from tabular.schemas import TabularResult  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def build_result(article: NMLArticle) -> TabularResult:
    """Compute TabularResult for one NMLArticle.

    Aggregates token classifications across all <pre> blocks in document
    order; <p> tokens are counted via the reader helper (already excludes
    trailers).
    """
    p_tokens = count_p_body_tokens(article.p_blocks_text)
    pre_tab = 0
    pre_narr = 0
    for pre in article.pre_blocks_text:
        is_tabular, token_counts = classify_pre_block(pre)
        for flag, n in zip(is_tabular, token_counts):
            if flag:
                pre_tab += n
            else:
                pre_narr += n
    return TabularResult(
        accession_number=article.accession_number,
        p_tokens=p_tokens,
        pre_tabular_tokens=pre_tab,
        pre_narrative_tokens=pre_narr,
    )


def write_sidecar(articles: Iterable[NMLArticle], output: Path) -> int:
    """Write one TabularResult JSONL line per article. Returns count written.

    Creates parent directories if needed. Overwrites any existing file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output, "w") as f:
        for art in articles:
            result = build_result(art)
            f.write(result.model_dump_json() + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nml-file", type=Path, required=True,
                    help="Path to a raw monthly DJN NML file.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output sidecar JSONL path.")
    args = ap.parse_args()

    if not args.nml_file.exists():
        raise FileNotFoundError(f"NML file not found: {args.nml_file}")

    logger.info("Reading %s", args.nml_file)
    articles = iter_nml_articles(args.nml_file)
    n = write_sidecar(articles, args.output)
    logger.info("Wrote %d TabularResult rows to %s", n, args.output)


if __name__ == "__main__":
    main()
