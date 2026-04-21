# DJNW dataset notes

Issues we've run into with the Dow Jones Newswires dump at
`/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/` and how we handle them.

## Files

- Monthly chunks named `YYYY-MM{a,b,c,d}_clean.jsonl` (not `YYYY-MM_clean.jsonl`).
  Our loader globs `*_clean.jsonl` and uses the first 7 chars (`YYYY-MM`) for
  date range filtering, so the `a/b/c/d` suffixes are fine.
- Per-article JSON keys: `accession_number`, `docdate`, `display_date`,
  `headline`, `text`, `codes`, `product`, `sequence`. **No `url` field** — DJNW
  is proprietary archive data with no public permalinks. The metadata
  directory one level up (`/.../metadata/`) contains firm-linkage tables
  (`article_firm_associations.csv`, `by_permno/`, etc.), not URLs.

## Text shapes

DJNW `text` arrives in three very different shapes, and the paragraph splitter
has to handle all three:

1. **Short run-on articles** (median ~290 chars) — zero newlines, a few
   sentences. Many early articles in a month are correction notices or
   market-data blurbs of this form.
2. **Long pre-wrapped filings** (e.g. 30 KB HK Bourse announcements) — hundreds
   of single-`\n` line breaks at ~80 cols, with semantic paragraphs separated
   by `\n\n`. Splitting naively on `\n` produces one "paragraph" per display
   line.
3. **Mixed** — normal articles with proper `\n\n` paragraph breaks.

`experiments/loaders.py::split_into_paragraphs` now:

1. Prefers `\n\n` paragraph breaks when present, and **unwraps single-`\n` line
   wrapping** inside each block.
2. Otherwise unwraps the whole text into one block and groups sentences.

Before this fix a 100-article sample produced 77 single-paragraph articles,
19 zero-paragraph articles (silently skipped by the pipeline), and one article
with 85 paragraphs from a long filing. After: no zero-paragraph articles, long
filings land in a handful of semantic paragraphs.

## Content quality

Sorted by `accession_number`, the first hundred or so articles each month are
heavy on correction notices, HK Bourse filings, and ultra-short market-data
snippets — not substantive macro news. Any sampling strategy should filter by
`product` code or minimum text length before taking the top N. We haven't added
that filter yet; for early experiments we just run on the raw top-N and accept
that many will be junk.

## Loader call

```python
load_djnw_articles(
    data_dir=Path("/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles"),
    max_articles=100,
    start_date="2020-11",
    end_date="2020-11",
)
```

Returns the standard article dict: `{id, headline, paragraphs, date, source, codes}`.
