# autoresearch

Automated prompt optimization for a macro news → futures mapping pipeline.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr7`). Each new day gets a new branch to separate experiments.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current best branch (or main if starting fresh).
3. **Read the in-scope files**: Read these files for full context:
   - `autoresearch/program.md` — this file. Your instructions.
   - `autoresearch/evaluate.py` — fixed evaluation script. Do not modify.
   - `autoresearch/regression_tests.yaml` — fixed regression tests. Do not modify.
   - `src/mapping/prompts/single_asset.txt` — ArticleMapper prompt (per-asset). **You modify this.**
   - `src/mapping/prompts/single_asset_paragraph.txt` — ParagraphMapper prompt (per-asset). **You modify this.**
   - `src/mapping/prompts/summarize.txt` — Macro summary prompt (Stage 0). **You modify this.**
   - `src/mapping/prompts/validate.txt` — Validator prompt. **You modify this.**
4. **Verify model exists**: Check that `~/models/gemma-4-31b-it` exists. If not, tell the human.
5. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Seed best prompts**: Copy current prompts to `autoresearch/best/`:
   ```
   mkdir -p autoresearch/best
   cp src/mapping/prompts/single_asset.txt autoresearch/best/
   cp src/mapping/prompts/single_asset_paragraph.txt autoresearch/best/
   cp src/mapping/prompts/summarize.txt autoresearch/best/
   cp src/mapping/prompts/validate.txt autoresearch/best/
   ```
7. **Confirm and go**: Confirm setup looks good, then kick off.

## Pipeline Overview

This is a four-stage per-asset LLM pipeline that maps news articles to global futures contracts. Each (article, asset) and (paragraph, asset) pair gets its own independent LLM call — this eliminates copy-paste reasoning between assets.

1. **Summarize** (`summarize.txt`) — reads full articles, generates macro summary (1 call per article). Output passed as `[CONTEXT]` to ParagraphMapper.
2. **ArticleMapper** (`single_asset.txt`) — evaluates each (article × asset) pair independently (95 calls per article). Returns `{relevant, signal, reasoning}`.
3. **ParagraphMapper** (`single_asset_paragraph.txt`) — evaluates each (paragraph × asset) pair independently with macro summary context (95 calls per paragraph). Returns `{relevant, signal, reasoning}`.
4. **Validator** (`validate.txt`) — accepts or rejects each proposed mapping from stages 2 and 3 (1 call per proposed pair).

The union of ArticleMapper and ParagraphMapper proposals goes to the Validator. An asset can be flagged by AM only, PM only, or both — the Validator sees reasoning from whichever mapper(s) proposed it.

The pipeline runs on a SLURM GPU cluster using vLLM with Gemma 4 31B IT. Prefix caching makes the per-asset approach efficient: the article text prefix is identical across all 95 asset calls.

## What You CAN Do

- Modify `src/mapping/prompts/single_asset.txt` — ArticleMapper prompt.
- Modify `src/mapping/prompts/single_asset_paragraph.txt` — ParagraphMapper prompt.
- Modify `src/mapping/prompts/summarize.txt` — Macro summary prompt.
- Modify `src/mapping/prompts/validate.txt` — Validator prompt.

## What You CANNOT Do

- Modify `autoresearch/evaluate.py`. It is read-only. It contains the fixed evaluation metric.
- Modify `autoresearch/regression_tests.yaml`. It is read-only.
- Modify any Python source code in `src/`. Only prompt `.txt` files are in scope.
- Install new packages or add dependencies.
- Modify the `OUTPUT FORMAT` section of any prompt (breaks JSON parsing).
- Change STEP numbering (STEP 0, STEP 1, STEP 2, etc.) in any prompt.

## The Goal

**Maximize the composite evaluation score** while passing all hard constraints and regression tests.

The composite score (higher is better) is a weighted combination of:
| Component | Weight | Description |
|-----------|--------|-------------|
| acceptance_rate | 0.25 | total_accepted / total_pairs on gold dataset |
| 1 - fpr | 0.25 | 1 - false_positive_rate on gold dataset |
| article_only_acceptance | 0.15 | acceptance rate for assets flagged only by ArticleMapper |
| paragraph_only_acceptance | 0.15 | acceptance rate for assets flagged only by ParagraphMapper |
| coverage_breadth | 0.20 | unique accepted assets / 95 (universe size) |

**Hard constraints** (auto-reject if any fail):
1. Strong signal acceptance rate > 90%
2. No empty validator reasonings

**Regression tests**: Curated (article, asset, expected_outcome) tuples in `regression_tests.yaml`. All must pass.

## Known Weaknesses (starting point for ideas)

Based on v2 gold results (346 accepted, 129 rejected, 475 total pairs):

- **Overall FPR 27.2%** — too many false positives getting rejected by Validator.
- **pm_weak acceptance 61.2%** — ParagraphMapper weak signals are the main noise source. AM+PM `both` source has 94.2% acceptance.
- **article_only acceptance ~65%** — assets flagged only by ArticleMapper often get rejected by Validator. Improving AM reasoning quality would help.
- **paragraph_only volume**: PM generates many solo proposals (164 paragraph_only), with 29.3% acceptance. These have value (some good finds AM misses) but are noisy.
- **Company-specific leakage**: gold_04 and gold_08 are company-specific articles but ParagraphMapper still proposes some assets. The ArticleMapper correctly returns nothing for these.
- **Commodity over-tagging** — mappers over-tag commodities without clear supply/demand signals.
- **Equity over-tagging** — mappers over-tag equity indices on generic risk sentiment.

## Experiment Execution

Each experiment requires ONE SLURM GPU job (gold dataset only), followed by evaluation.

### Submitting the gold experiment:

Use the launcher script. Name the output JSON with the short commit hash so every iteration is preserved:

```bash
COMMIT=$(git rev-parse --short HEAD)
bash scripts/article_depth_experiment.sh --dataset gold --output-json autoresearch/gold_${COMMIT}.json
```

### Running evaluation (after gold finishes):

```bash
COMMIT=$(git rev-parse --short HEAD)
source .venv/bin/activate && \
python autoresearch/evaluate.py \
    --gold autoresearch/gold_${COMMIT}.json \
    --regression autoresearch/regression_tests.yaml \
    --output autoresearch/eval_${COMMIT}.json
```

### Monitoring jobs:

```bash
squeue -u $USER
```

Poll with `squeue` to check if jobs are done. A job is done when it no longer appears in the queue. Then read the output files to check for errors before running evaluation.

**Typical timing**: gold takes ~25 minutes (15 articles × 95 assets per article/paragraph + validation).

## Output Format

The evaluation script prints a summary and writes `eval_report.json`. Key fields:
- `passed`: true/false (hard constraints + regression tests)
- `composite_score`: the metric to maximize
- `components`: breakdown of each weighted component
- `hard_constraints.violations`: list of constraint failures
- `regression_tests.failures`: list of regression failures

## Logging Results

When an experiment is done, log it to `autoresearch/results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	score	passed	status	description
```

1. git commit hash (short, 7 chars)
2. composite_score (e.g. 0.7280) — use 0.0000 for crashes
3. passed: true/false (hard constraints + regression tests)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	score	passed	status	description
a1b2c3d	0.7280	true	keep	baseline
b2c3d4e	0.7350	true	keep	tighten commodity reasoning in single_asset.txt
c3d4e5f	0.7100	false	discard	failed regression: gold_01/Canadian Dollar lost
d4e5f6g	0.0000	false	crash	SLURM job OOM
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr7`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Make ONE focused change to one or more prompt files. Keep diffs small (<20 lines). Write a clear commit message describing your hypothesis.
3. `git commit` the prompt change.
4. Submit the gold SLURM job. Wait for it to finish by polling `squeue`.
5. Check for errors: read the `.err` files. If the job crashed, read the error, attempt a fix if trivial, otherwise log as `crash` and revert.
6. Run `evaluate.py` to compute the composite score.
7. Record the results in `results.tsv` (NOTE: do not commit results.tsv, leave it untracked by git).
8. **If composite_score improved AND passed is true**: KEEP. You "advance" the branch. Also update `autoresearch/best/` with the new prompts and eval_report.json.
9. **If composite_score is equal or worse, OR passed is false**: DISCARD. `git reset --hard HEAD~1` to revert the prompt change. Restore prompts from `autoresearch/best/`.

**Simplicity criterion**: All else being equal, simpler prompts are better. A tiny score improvement that adds ugly complexity to the prompt is not worth it. Removing prompt text and getting equal or better results is a great outcome.

**Crashes**: If a SLURM job crashes (OOM, bug, etc.), use your judgment. If it's a cluster issue (node down, preemption), just resubmit. If it's caused by your prompt change somehow, revert and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the prompts for new angles, try combining previous near-misses, try more radical prompt restructuring. The loop runs until the human interrupts you, period.
