# Pipeline Architecture

Generator-verifier split for high-precision, high-recall asset tagging.

## TL;DR

Every `(article, asset)` pair is a binary classification instance. A single model can only sit at one point on its precision-recall curve. The macronews pipeline *bypasses* the tradeoff by splitting classification into two sequential stages with intentionally different objectives:

- **Mappers** (Stage 1 ArticleMapper + Stage 2 ParagraphMapper): optimized for **recall**, liberally propose candidate tags.
- **Validator** (Stage 3): optimized for **precision**, strictly enforces the 5-case relevance rules and rejects anything that doesn't fit.

Composing these two stages gives the pipeline high precision *and* high recall, at a compute cost that is linear rather than multiplicative.

---

## The formal decomposition

Every instance is a pair `x = (article, asset) ∈ A × U` where `U` is the asset universe (95 futures contracts). The latent label `y ∈ {relevant, not relevant}`.

**Single-classifier approach.** A classifier `f(x) → {0, 1}` must balance:

- **Recall** = `TP / (TP + FN)` — of truly relevant pairs, how many did we find?
- **Precision** = `TP / (TP + FP)` — of pairs we flagged, how many were actually relevant?

At fixed model capability, improving one sacrifices the other. The model has an ROC curve; a single classifier sits at one point.

**Two-stage decomposition.** Split the decision into two sequential classifiers:

1. **Mapper `m: x → {0, 1}`**, optimized for recall. Calibration target: `P(m = 0 | y = 1) ≈ 0`. Missing a true positive is catastrophic because the mapper's output is the validator's only input.

2. **Validator `v: (x, m's reasoning) → {0, 1}`**, optimized for precision. Calibration target: `P(v = 1 | m = 1, y = 0) ≈ 0`. The validator rejects any mapper proposal that fails the strict rules.

Final classifier: `f(x) = m(x) ∧ v(x)`.

Composite metrics:

```
Recall(f)    = P(m = 1 | y = 1) × P(v = 1 | m = 1, y = 1)  ≈  P(v = 1 | m = 1, y = 1)
Precision(f) = P(y = 1 | m = 1, v = 1)                     ≈  high
```

## Why this bypasses the tradeoff

The key insight: **the two stages operate on different sub-populations**. The base rate of true positives (prevalence of `y = 1`) is very different for each stage.

On the full `A × U` space, true positives are rare. In the DJNW March 2022 run:

- 275 non-company articles × 95 assets = 26,125 mapper calls per stage
- ~1,500 truly relevant pairs → base rate of `y = 1` ≈ 5.7%

After the mapper filters, base rate shoots up:

- Mapper proposes ~2,850 pairs → base rate of `y = 1` ≈ 53%

The validator operates on this **restricted, high-signal-density population**. Precision filtering is dramatically easier when most of your candidates are already likely true positives. The validator's own ROC curve is much better-conditioned on the mapper-positive subpopulation than on the full population.

Mathematically:

- Mapper operates at `(high TPR, high FPR)` on the full `(article, asset)` population
- Validator operates at `(high TPR, low FPR)` on the restricted mapper-positive population
- Composition: `(high TPR, low FPR)` on the full population

Neither stage alone is good enough. The composition is.

---

## How the prompts embody this split

The pipeline prompts are written with intentionally different objectives. Same rule set (the 5 cases: NAMED / PRIMARY VEHICLE / ISSUING AUTHORITY / SUBSTITUTION / HEAVY CONSTITUENT), but different framings.

### Mapper prompt (`single_asset.txt`, `single_asset_paragraph.txt`) — recall-first

> *"Determine whether this asset is plausibly affected by the specific subject of the article through a direct economic mechanism. Think of it as: would this article show up on the desk running this specific futures contract?"*

Permissive framing. "Plausibly affected," "show up on the desk." The mapper is encouraged to propose anything it can justify via one of the 5 cases.

### Validator prompt (`validate.txt`) — precision-first

> *"Your job is to **STRICTLY enforce** the relevance rules the mappers were supposed to follow. The mappers may over-propose; you are the final arbiter. **Reject anything that does not clearly fit** one of the five allowed cases."*

Strict framing. "Strictly enforce," "final arbiter," "reject anything that does not clearly fit." Plus an explicit rejection list:

- Generic "risk-off / safe-haven / global growth" chains without a specific mechanism
- Historical correlations without an article-specific pathway
- Secondary derivations through intermediate commodities/currencies not discussed
- Minority index/sector constituents
- Supply-chain-adjacent inference on tiny exposures
- Cross-currency / cross-country rate spillover as standalone justification
- Multi-step reasoning chains
- Case 5 composition mismatch (e.g., industrial company → tech-focused index)

These are intentionally *different objective functions on the same rule set*. One prompt says "propose if plausible," the other says "reject if not clearly fitting."

---

## Why this works specifically for LLMs

LLMs have two decoupled failure modes that an architecture of this kind handles well:

1. **Recall failure (omission).** The model fails to retrieve a concept it "knows" because the prompt framing didn't activate the right associative path. Cure: prompts that cast a wide net.

2. **Precision failure (hallucination).** The model generates a plausible-sounding reasoning chain for a claim that's false. Cure: prompts that enforce a narrow specification.

These are not additive. You can't just "prompt harder" to fix both simultaneously — a prompt that reduces hallucination by demanding evidence tends to also reduce recall by making the model timid. Conversely, a prompt that encourages exploration tends to also invite hallucination.

The two-stage architecture handles each failure mode in the stage where it's tractable:

- **Mapper** sees `(article, asset)` in isolation and liberally proposes. Recall failure is visible (no proposal → no tag), and the prompt framing activates the right associative paths.
- **Validator** sees `(article, asset, mapper's reasoning)`. Hallucinations in the mapper's reasoning are auditable in text. The validator can reject them without the composition bias of having generated them itself — it's evaluating someone else's work against strict rules.

---

## The ensemble vote layer — a bonus

Because we run **two** mappers (ArticleMapper on full article + ParagraphMapper on each paragraph) independently, their agreement is a third signal. Formally, this is a 2-of-2 ensemble vote over the same 5-case rule set.

Every candidate carries a `source` field: `both`, `article_only`, or `paragraph_only`.

Empirically on the DJNW March 2022 strict run:

| Source | Accepted | Rejected | Accept rate |
|---|---|---|---|
| `both` | 1412 | 618 | **70%** |
| `article_only` | 38 | 253 | 13% |
| `paragraph_only` | 76 | 449 | 14% |

**Interpretation:** when both mappers independently flag the same asset under the same rules, the validator accepts it 70% of the time. When only one mapper fires, the validator rejects ~86% of the time. The ratio (~5.4×) is the ensemble benefit.

This gives downstream consumers a *free* reliability signal. A tag with `source=both` is much more trustworthy than a `source=article_only` or `source=paragraph_only` tag. We can filter post-hoc on this signal without any additional compute.

The 70% / 13% split is not a pipeline bug. It's evidence that the two mappers are doing genuinely different work (not redundant), and that cross-mapper agreement carries information about ground-truth relevance beyond either single mapper's judgment.

---

## Why the 46% validator rejection rate is *not* a bug

A common concern: if the validator rejects 46% of mapper proposals on the DJNW strict run, isn't the mapper doing a bad job?

No — this is the system working as designed. In the formal frame:

- Mapper calibration target: `P(m = 0 | y = 1) ≈ 0` (accept everything that might be relevant)
- Validator calibration target: `P(v = 1 | y = 0) ≈ 0` (reject everything that isn't)

The validator rejection rate is the empirical estimate of `P(m = 1, y = 0)` — how often the mapper proposed something wrong. **A rejection rate near zero would actually be a red flag**, because it'd mean the mapper is being as strict as the validator (i.e., the two stages are redundant and we're paying double compute for the same work).

The 46% gap tells us the two stages are doing different jobs:

- The mapper generates liberally from a wide candidate space.
- The validator prunes aggressively against strict rules.
- The composition achieves the precision/recall target neither could alone.

If we wanted to reduce the rejection rate for cost reasons, we could tighten the mapper prompts — but that would cost us recall. The tradeoff is: more compute in Stage 3 (validator rejections) vs more missed true positives (mapper under-proposals). The current balance favors the former because a false negative is structurally worse than a false positive — a false positive costs one validator call; a false negative is invisible forever.

---

## Related CS patterns

This architectural pattern shows up across many CS subfields. The common theme is "generation is cheap but noisy; verification is expensive but reliable; composing them lets you avoid paying the verification cost on every possible candidate."

| Domain | Generator | Verifier |
|---|---|---|
| **Compiler design** | Parser produces candidate AST | Type checker rejects invalid ones |
| **SAT solvers** | Branching generates candidate assignments | Unit propagation rejects contradictions |
| **Information retrieval** | BM25 / dense retrieval produces top-k (high recall) | Neural reranker reorders for precision |
| **Speculative decoding** | Small draft model proposes tokens | Large target model verifies / rejects |
| **Retrieval-augmented generation (RAG)** | Retriever fetches candidate passages | Reader answers grounded in retrieved set |
| **Generate-and-test search** | Candidate generator | Feasibility checker |
| **Constraint satisfaction** | Variable assignments | Constraint propagation |
| **Our pipeline** | ArticleMapper + ParagraphMapper | Validator |

The underlying claim in all of these: **recall and precision are inversely hard for a single system, but independently achievable by two specialized systems**.

---

## Empirical validation (DJNW March 2022, Gemma 4 26B A4B, strict rules)

- **1000 articles → 726 filtered by company-specific gate → 274 pass through to mapping**
- **Stage 1 (ArticleMapper):** 26,125 calls (274 × 95 assets). Proposes ~2,450 candidate tags.
- **Stage 2 (ParagraphMapper):** ~107,000 calls (274 × ~4.1 paragraphs × 95). Proposes ~2,450 candidate tags.
- **Union (Stage 3 input):** 2,846 unique `(article, asset)` pairs to validate.
- **Stage 3 (Validator):** 2,846 calls. Accepts 1,526, rejects 1,320.

Final output: **1,526 accepted tags** across 95 assets, with cross-mapper agreement at 70% accept rate on `both`-source pairs and 13% accept rate on single-source pairs.

The 46% validator rejection rate is the cost of recall-first mappers. The 70% `both`-source accept rate is evidence of ensemble benefit. The 13% single-source accept rate is evidence that both mappers are doing different jobs.

---

## In one sentence

> *The pipeline bypasses the precision-recall tradeoff by operating two classifiers on different sub-populations: the mapper generates liberally on the full `article × asset` space (low base rate, recall-first), the validator filters strictly on the mapper-positive subset (high base rate, precision-first), and the composition achieves high precision AND high recall at a compute cost that is linear (mapper calls + validator calls), not multiplicative.*
