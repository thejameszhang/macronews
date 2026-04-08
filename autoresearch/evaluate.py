#!/usr/bin/env python3
"""
Composite metric evaluator for the autoresearch prompt optimization loop.

Reads gold_summary.json + regression_tests.yaml.
Outputs eval_report.json with hard constraints, regression tests, and composite score.

Usage:
    python autoresearch/evaluate.py \
        --gold results/gold_summary.json \
        --regression autoresearch/regression_tests.yaml \
        --output autoresearch/eval_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------

def check_hard_constraints(gold: dict) -> tuple[bool, list[str]]:
    violations: list[str] = []
    agg = gold["aggregate"]

    # 1. Strong signal acceptance rate > 90%
    am_strong = agg.get("by_mapper_signal", {}).get("am_strong", {})
    if am_strong.get("total", 0) > 0:
        strong_rate = am_strong["acceptance_rate"]
        if strong_rate < 0.90:
            violations.append(
                f"Strong signal acceptance rate {strong_rate:.3f} < 90%"
            )
    # If no strong signals exist, that's also a problem
    elif agg["total_pairs"] > 0:
        violations.append("No strong signals found in gold results")

    # 2. No empty validator reasonings
    for art in gold["articles"]:
        for mapping in art.get("accepted", []) + art.get("rejected", []):
            vr = mapping.get("validator_reasoning", "")
            if not vr or not vr.strip():
                violations.append(
                    f"Empty validator reasoning: {art['article_id']} / "
                    f"{mapping['asset']}"
                )

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

def _build_lookup(data: dict) -> dict[str, dict[str, str]]:
    """Build {article_id: {asset_name: 'accepted'|'rejected'}} from result JSON."""
    lookup: dict[str, dict[str, str]] = {}
    for art in data["articles"]:
        aid = art["article_id"]
        lookup[aid] = {}
        for m in art.get("accepted", []):
            lookup[aid][m["asset"]] = "accepted"
        for m in art.get("rejected", []):
            lookup[aid][m["asset"]] = "rejected"
    return lookup


def check_regression_tests(
    gold: dict, tests_path: Path
) -> tuple[bool, list[str]]:
    if not tests_path.exists():
        return True, []

    with open(tests_path) as f:
        tests_data = yaml.safe_load(f)

    tests = tests_data.get("tests", [])
    if not tests:
        return True, []

    datasets = {"gold": gold}
    failures: list[str] = []

    for test in tests:
        ds_name = test["dataset"]
        ds = datasets.get(ds_name)
        if ds is None:
            failures.append(f"Unknown dataset: {ds_name}")
            continue

        lookup = _build_lookup(ds)
        article_id = test["article_id"]
        asset = test["asset"]
        expected = test["expected"]

        # Expand wildcards
        article_ids = (
            list(lookup.keys()) if article_id == "*" else [article_id]
        )

        for aid in article_ids:
            if aid not in lookup:
                if expected == "accepted":
                    failures.append(
                        f"[{ds_name}] {aid}/{asset}: article not found, "
                        f"expected {expected}"
                    )
                continue

            assets_to_check = (
                list(lookup[aid].keys()) if asset == "*" else [asset]
            )

            for a in assets_to_check:
                actual = lookup[aid].get(a)

                if expected == "accepted":
                    if actual != "accepted":
                        failures.append(
                            f"[{ds_name}] {aid}/{a}: expected accepted, "
                            f"got {actual or 'not proposed'}"
                        )
                elif expected == "rejected":
                    if actual == "accepted":
                        failures.append(
                            f"[{ds_name}] {aid}/{a}: expected rejected, "
                            f"got accepted"
                        )
                elif expected == "not_proposed":
                    if actual is not None:
                        failures.append(
                            f"[{ds_name}] {aid}/{a}: expected not_proposed, "
                            f"got {actual}"
                        )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

WEIGHTS = {
    "acceptance_rate": 0.25,
    "inv_fpr": 0.25,
    "article_only_acceptance": 0.15,
    "paragraph_only_acceptance": 0.15,
    "coverage_breadth": 0.20,
}

UNIVERSE_SIZE = 95  # 23 equity + 11 equity_sector + 9 currency + 25 commodity + 18 bond + 7 stir + 2 volatility


def compute_composite_score(
    gold: dict,
) -> tuple[float, dict[str, float], dict]:
    agg = gold["aggregate"]

    total_pairs = agg["total_pairs"]
    total_accepted = agg["total_accepted"]

    # Acceptance rate
    acceptance_rate = total_accepted / total_pairs if total_pairs > 0 else 0.0

    # 1 - FPR
    fpr = agg.get("false_positive_rate", 0.0)
    inv_fpr = 1.0 - fpr

    # Article-only acceptance rate
    ao = agg.get("by_source", {}).get("article_only", {})
    article_only_acceptance = ao.get("acceptance_rate", 0.0)

    # Paragraph-only acceptance rate
    po = agg.get("by_source", {}).get("paragraph_only", {})
    paragraph_only_acceptance = po.get("acceptance_rate", 0.0)

    # Coverage breadth: unique accepted assets / universe size
    unique_accepted = set()
    for art in gold["articles"]:
        for m in art.get("accepted", []):
            unique_accepted.add(m["asset"])
    coverage_breadth = len(unique_accepted) / UNIVERSE_SIZE

    components = {
        "acceptance_rate": acceptance_rate,
        "inv_fpr": inv_fpr,
        "article_only_acceptance": article_only_acceptance,
        "paragraph_only_acceptance": paragraph_only_acceptance,
        "coverage_breadth": coverage_breadth,
    }

    composite = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)

    raw = {
        "gold_accepted": total_accepted,
        "gold_rejected": agg["total_rejected"],
        "gold_pairs": total_pairs,
        "gold_fpr": fpr,
        "gold_unique_accepted": len(unique_accepted),
        "strong_acceptance_rate": (
            agg.get("by_mapper_signal", {})
            .get("am_strong", {})
            .get("acceptance_rate", 0.0)
        ),
    }

    return composite, components, raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate autoresearch iteration")
    parser.add_argument("--gold", required=True, help="Path to gold_summary.json")
    parser.add_argument(
        "--regression",
        default="autoresearch/regression_tests.yaml",
        help="Path to regression_tests.yaml",
    )
    parser.add_argument("--output", required=True, help="Path to write eval_report.json")
    args = parser.parse_args()

    gold = json.loads(Path(args.gold).read_text())

    hc_passed, hc_violations = check_hard_constraints(gold)
    rt_passed, rt_failures = check_regression_tests(
        gold, Path(args.regression)
    )
    composite, components, raw = compute_composite_score(gold)

    passed = hc_passed and rt_passed

    report = {
        "passed": passed,
        "hard_constraints": {
            "passed": hc_passed,
            "violations": hc_violations,
        },
        "regression_tests": {
            "passed": rt_passed,
            "failures": rt_failures,
        },
        "composite_score": round(composite, 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "weights": WEIGHTS,
        "raw": raw,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    # Print summary
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] composite={composite:.4f}")
    if hc_violations:
        for v in hc_violations:
            print(f"  HC VIOLATION: {v}")
    if rt_failures:
        for f in rt_failures:
            print(f"  REGRESSION: {f}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
