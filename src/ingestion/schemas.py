"""
ArticleRecord — the canonical output schema for all ingestion sources.

Every ingestor (DJNW, Reuters, WSJ) must produce Parquet files that conform
to this schema. Downstream stages (embedding, MIF training) read only this schema.
"""

import polars as pl


ARTICLE_SCHEMA = {
    "article_id":     pl.Utf8,                                      # hash(source + stable_id)
    "effective_date": pl.Date,                                      # trading day t assignment
    "effective_time": pl.Datetime("us", time_zone="America/New_York"),  # eff_time in ET
    "buffer_zone":    pl.Boolean,                                   # True if eff_time in [9:00, 9:30) AM ET
    "source":         pl.Utf8,                                      # "djnw" | "reuters" | "wsj"
    "headline":       pl.Utf8,
    "body_excerpt":   pl.Utf8,                                      # headline + body[:1200] (or headline-only)
    "subject_codes":  pl.List(pl.Utf8),                            # topic codes (DJCS / Reuters / WSJ section)
    "rics":           pl.List(pl.Utf8),                            # Reuters Instrument Codes (empty for WSJ)
    "word_count":     pl.Int32,
}

# Minimum word count for a record to be ingested.
# For headline-only mode (WSJ Phase 1), set to 5 rather than the 30-word
# threshold that applies once body text is available.
MIN_WORD_COUNT_HEADLINE_ONLY = 5
MIN_WORD_COUNT_WITH_BODY = 30
