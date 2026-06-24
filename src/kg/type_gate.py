"""Flag-and-drop type-signature gate (ex-post, non-lossy at corpus level).

Splits each event's triplets: type-valid ones stay on the event (cleaned sidecar =
the only downstream input); violating ones move to an audit-only rejected log.
"""
import argparse
import json
from collections import Counter

from kg.type_signatures import type_violation


def gate_article(article):
    """Return (cleaned_article, rejected_rows). Does not mutate the input."""
    aid = article.get("article_id")
    rejected = []
    cleaned_events = []
    for ev in (article.get("events") or []):
        kept = []
        for t in (ev.get("triplets") or []):
            reason = type_violation(t.get("subject_type"), t.get("relation"), t.get("object_type"))
            if reason is None:
                kept.append(t)
            else:
                rejected.append({
                    "article_id": aid,
                    "event_id": ev.get("id"),
                    "statement": ev.get("statement"),
                    "subject": t.get("subject"), "subject_type": t.get("subject_type"),
                    "relation": t.get("relation"),
                    "object": t.get("object"), "object_type": t.get("object_type"),
                    "value": t.get("value"),
                    "type_violation": reason,
                })
        new_ev = {**ev, "triplets": kept}
        cleaned_events.append(new_ev)
    cleaned = {**article, "events": cleaned_events}
    return cleaned, rejected


def main():
    ap = argparse.ArgumentParser(description="Ex-post type-signature gate.")
    ap.add_argument("events", help="extractor/event sidecar JSONL")
    ap.add_argument("--clean", required=True, help="cleaned sidecar out (downstream input)")
    ap.add_argument("--rejected", required=True, help="audit-only rejected-triplet log out")
    ap.add_argument("--summary", action="store_true", help="print per-relation drop counts")
    args = ap.parse_args()

    drops = Counter()
    kept_n = 0
    with open(args.events) as f, open(args.clean, "w") as out, open(args.rejected, "w") as rej:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cleaned, rejected = gate_article(json.loads(line))
            out.write(json.dumps(cleaned) + "\n")
            for r in rejected:
                rej.write(json.dumps(r) + "\n")
                drops[r["relation"]] += 1
            kept_n += sum(len(ev.get("triplets") or []) for ev in cleaned["events"])

    total = sum(drops.values())
    print(f"kept {kept_n:,} triplets; dropped {total:,}")
    if args.summary:
        for r, c in drops.most_common():
            print(f"  {r:<18}{c:>7,}")


if __name__ == "__main__":
    main()
