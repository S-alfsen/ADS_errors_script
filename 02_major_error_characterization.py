#!/usr/bin/env python3
"""Characterize major-error forms by consensus severity, taxonomy, and origin."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from publication_utils import count_pct, median, pct, quantile, read_csv, require_columns, write_csv, yes


SEVERITY_LABELS = {
    0: "Indeterminate",
    1: "Low potential harm",
    2: "Moderate potential harm",
    3: "Considerable potential harm",
    4: "Severe potential harm",
    5: "Life-threatening potential harm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encounters", required=True, help="Main-cohort encounter-level CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for aggregate outputs.")
    parser.add_argument("--major-error-column", default="major_error_flag")
    parser.add_argument("--major-error-count-column", default="major_error_count")
    parser.add_argument("--severity-column", default="major_error_consensus_severity_score")
    parser.add_argument("--taxonomy-column", default="major_error_categories")
    parser.add_argument("--origin-column", default="resolved_error_origin")
    parser.add_argument("--delimiter", default=";")
    return parser.parse_args()


def canonical_taxonomy(token: str) -> str:
    text = token.strip().lower()
    if not text:
        return "Other"
    if "utelatelse" in text or "omission" in text:
        return "Omission"
    if "negasjon" in text or "negation" in text:
        return "Negation error"
    if "faktafeil" in text or "factual" in text:
        return "Factual"
    if "hallus" in text or "tillegg" in text or "addition" in text or "hallucin" in text:
        return "Hallucination / Addition"
    return "Other"


def canonical_origin(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return "Missing"
    if text in {"already in transcript", "transcript", "transcript / text-to-speech", "transcript / speech-to-text"}:
        return "Transcript"
    if text in {"only in generated note", "generated note", "generated note / llm", "draft note"}:
        return "Draft note"
    if text in {"both", "both transcript and llm", "both transcript/draft note"}:
        return "Both"
    if text in {"unclear", "uncertain", "usikker"}:
        return "Uncertain"
    if "transcript" in text and "llm" in text:
        return "Both"
    if "transcript" in text or "speech" in text:
        return "Transcript"
    if "generated" in text or "draft" in text or "llm" in text:
        return "Draft note"
    return "Uncertain"


def parse_score(value: str) -> int:
    raw = (value or "").strip()
    if raw == "":
        raise ValueError("Missing consensus severity score in a major-error row")
    score = int(float(raw.replace(",", ".")))
    if score not in SEVERITY_LABELS:
        raise ValueError(f"Unexpected severity score: {value!r}")
    return score


def main() -> int:
    args = parse_args()
    rows = read_csv(args.encounters, args.delimiter)
    require_columns(
        rows,
        [
            args.major_error_column,
            args.major_error_count_column,
            args.severity_column,
            args.taxonomy_column,
            args.origin_column,
        ],
        "encounters",
    )
    major_rows = [row for row in rows if yes(row[args.major_error_column])]
    denominator = len(major_rows)
    if denominator == 0:
        raise ValueError("No major-error rows found")

    severity_counts = Counter(parse_score(row[args.severity_column]) for row in major_rows)
    severity_rows = [
        {
            "severity_score": score,
            "severity_label": label,
            "forms": severity_counts.get(score, 0),
            "denominator": denominator,
            "percent": round(pct(severity_counts.get(score, 0), denominator), 1),
            "display": count_pct(severity_counts.get(score, 0), denominator),
        }
        for score, label in SEVERITY_LABELS.items()
    ]

    taxonomy_counts: Counter[str] = Counter()
    multi_label = 0
    for row in major_rows:
        raw_tokens = [token.strip() for token in (row[args.taxonomy_column] or "").split(";") if token.strip()]
        labels = sorted({canonical_taxonomy(token) for token in raw_tokens}) or ["Other"]
        if len(labels) > 1:
            multi_label += 1
        for label in labels:
            taxonomy_counts[label] += 1
    taxonomy_rows = [
        {
            "taxonomy": label,
            "forms": taxonomy_counts.get(label, 0),
            "denominator": denominator,
            "percent": round(pct(taxonomy_counts.get(label, 0), denominator), 1),
            "display": count_pct(taxonomy_counts.get(label, 0), denominator),
            "note": "Categories are not mutually exclusive",
        }
        for label in ["Factual", "Hallucination / Addition", "Omission", "Negation error", "Other"]
    ]
    taxonomy_rows.append(
        {
            "taxonomy": "Forms with multiple taxonomy categories",
            "forms": multi_label,
            "denominator": denominator,
            "percent": round(pct(multi_label, denominator), 1),
            "display": count_pct(multi_label, denominator),
            "note": "",
        }
    )

    origin_counts = Counter(canonical_origin(row[args.origin_column]) for row in major_rows)
    origin_rows = [
        {
            "origin": label,
            "forms": origin_counts.get(label, 0),
            "denominator": denominator,
            "percent": round(pct(origin_counts.get(label, 0), denominator), 1),
            "display": count_pct(origin_counts.get(label, 0), denominator),
        }
        for label in ["Transcript", "Draft note", "Both", "Uncertain", "Missing"]
    ]

    numeric_error_counts = []
    missing_numeric_count = 0
    for row in major_rows:
        raw = (row[args.major_error_count_column] or "").strip()
        if raw:
            numeric_error_counts.append(int(float(raw.replace(",", "."))))
        else:
            missing_numeric_count += 1
    total_reported_errors_minimum = sum(numeric_error_counts) + missing_numeric_count
    if numeric_error_counts:
        float_counts = [float(value) for value in numeric_error_counts]
        median_count = median(float_counts)
        q1_count = quantile(float_counts, 0.25)
        q3_count = quantile(float_counts, 0.75)
    else:
        median_count = q1_count = q3_count = None
    count_rows = [
        {
            "measure": "Total reported major errors",
            "value": total_reported_errors_minimum,
            "note": "Forms with missing numeric counts are counted as one major error.",
        },
        {
            "measure": "Forms with missing numeric major-error count",
            "value": missing_numeric_count,
            "note": "",
        },
        {
            "measure": "Reported major errors per major-error form, median",
            "value": median_count,
            "note": "Among forms with a nonmissing numeric count.",
        },
        {
            "measure": "Reported major errors per major-error form, first quartile",
            "value": q1_count,
            "note": "Among forms with a nonmissing numeric count.",
        },
        {
            "measure": "Reported major errors per major-error form, third quartile",
            "value": q3_count,
            "note": "Among forms with a nonmissing numeric count.",
        },
    ]

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "major_error_severity_summary.csv", severity_rows, args.delimiter)
    write_csv(output_dir / "major_error_taxonomy_summary.csv", taxonomy_rows, args.delimiter)
    write_csv(output_dir / "major_error_origin_summary.csv", origin_rows, args.delimiter)
    write_csv(output_dir / "major_error_count_summary.csv", count_rows, args.delimiter)
    print(f"Wrote aggregate outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
