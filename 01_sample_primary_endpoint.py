#!/usr/bin/env python3
"""Sample size, form coverage, primary endpoint, and figure-ready counts."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from publication_utils import count_pct, pct, read_csv, require_columns, safe_rate, wilson_ci, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encounters", required=True, help="Main-cohort encounter-level CSV.")
    parser.add_argument("--eligible-notes", required=True, help="Clinician-level eligible ADS-note counts CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for aggregate outputs.")
    parser.add_argument("--clinician-id-column", default="clinician_id")
    parser.add_argument("--specialty-column", default="specialty")
    parser.add_argument("--major-error-column", default="major_error_flag")
    parser.add_argument("--minor-error-column", default="minor_error_flag")
    parser.add_argument("--quality-column", default="note_quality")
    parser.add_argument("--eligible-note-count-column", default="eligible_ads_notes")
    parser.add_argument("--planning-rate", type=float, default=0.10)
    parser.add_argument("--planning-half-width", type=float, default=0.05)
    parser.add_argument("--delimiter", default=";")
    return parser.parse_args()


def yes(value: str) -> bool:
    return (value or "").strip().lower() in {"ja", "yes", "true", "1"}


def quality_label(value: str) -> str:
    raw = (value or "").strip()
    return {
        "Svært god": "Very good",
        "God": "Good",
        "Verken god eller dårlig": "Neutral",
        "Dårlig": "Poor",
        "Svært dårlig": "Very poor",
    }.get(raw, raw or "Missing")


def summarize_binary(label: str, count: int, denominator: int, note: str = "") -> dict[str, object]:
    low, high = wilson_ci(count, denominator)
    return {
        "measure": label,
        "count": count,
        "denominator": denominator,
        "percent": round(pct(count, denominator), 1),
        "wilson_ci95_low_percent": round(low * 100, 1),
        "wilson_ci95_high_percent": round(high * 100, 1),
        "display": count_pct(count, denominator),
        "note": note,
    }


def main() -> int:
    args = parse_args()
    encounters = read_csv(args.encounters, args.delimiter)
    eligible = read_csv(args.eligible_notes, args.delimiter)
    require_columns(
        encounters,
        [
            args.clinician_id_column,
            args.specialty_column,
            args.major_error_column,
            args.minor_error_column,
            args.quality_column,
        ],
        "encounters",
    )
    require_columns(eligible, [args.clinician_id_column, args.eligible_note_count_column], "eligible-notes")

    total_forms = len(encounters)
    clinicians = sorted({row[args.clinician_id_column] for row in encounters if row[args.clinician_id_column].strip()})
    eligible_notes = sum(int(float(row[args.eligible_note_count_column] or 0)) for row in eligible)
    sample_size = math.ceil((1.959963984540054**2) * args.planning_rate * (1 - args.planning_rate) / (args.planning_half_width**2))

    major_n = sum(yes(row[args.major_error_column]) for row in encounters)
    minor_n = sum(yes(row[args.minor_error_column]) for row in encounters)
    both_n = sum(yes(row[args.major_error_column]) and yes(row[args.minor_error_column]) for row in encounters)
    major_only_n = sum(yes(row[args.major_error_column]) and not yes(row[args.minor_error_column]) for row in encounters)
    minor_only_n = sum(yes(row[args.minor_error_column]) and not yes(row[args.major_error_column]) for row in encounters)
    no_error_n = total_forms - both_n - major_only_n - minor_only_n

    summary_rows = [
        {
            "measure": "Sample-size planning minimum notes",
            "count": sample_size,
            "denominator": "",
            "percent": "",
            "wilson_ci95_low_percent": "",
            "wilson_ci95_high_percent": "",
            "display": str(sample_size),
            "note": f"Normal approximation with assumed rate {args.planning_rate:g} and half-width {args.planning_half_width:g}.",
        },
        {
            "measure": "Clinicians",
            "count": len(clinicians),
            "denominator": "",
            "percent": "",
            "wilson_ci95_low_percent": "",
            "wilson_ci95_high_percent": "",
            "display": str(len(clinicians)),
            "note": "",
        },
        {
            "measure": "Submitted post-encounter forms",
            "count": total_forms,
            "denominator": "",
            "percent": "",
            "wilson_ci95_low_percent": "",
            "wilson_ci95_high_percent": "",
            "display": str(total_forms),
            "note": "",
        },
        {
            "measure": "Eligible ADS notes",
            "count": eligible_notes,
            "denominator": "",
            "percent": "",
            "wilson_ci95_low_percent": "",
            "wilson_ci95_high_percent": "",
            "display": str(eligible_notes),
            "note": "",
        },
        summarize_binary("Form coverage among eligible ADS notes", total_forms, eligible_notes),
        summarize_binary("Any major error", major_n, total_forms, "Primary endpoint"),
        summarize_binary("Any minor error", minor_n, total_forms),
        summarize_binary("No errors reported", no_error_n, total_forms),
        summarize_binary("Major errors only", major_only_n, total_forms),
        summarize_binary("Minor errors only", minor_only_n, total_forms),
        summarize_binary("Both minor and major errors", both_n, total_forms),
    ]

    specialty_rows = []
    specialty_counts: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in encounters:
        specialty_counts[row[args.specialty_column]].append(row)
    for specialty, rows in sorted(specialty_counts.items()):
        n = len(rows)
        major = sum(yes(row[args.major_error_column]) for row in rows)
        low, high = wilson_ci(major, n)
        specialty_rows.append(
            {
                "specialty": specialty,
                "forms": n,
                "major_error_forms": major,
                "major_error_percent": round(pct(major, n), 1),
                "wilson_ci95_low_percent": round(low * 100, 1),
                "wilson_ci95_high_percent": round(high * 100, 1),
                "display": count_pct(major, n),
            }
        )

    quality_counts = Counter(quality_label(row[args.quality_column]) for row in encounters)
    quality_rows = [
        {
            "quality_rating": label,
            "forms": quality_counts.get(label, 0),
            "denominator": total_forms,
            "percent": round(pct(quality_counts.get(label, 0), total_forms), 1),
        }
        for label in ["Very good", "Good", "Neutral", "Poor", "Very poor", "Missing"]
        if quality_counts.get(label, 0)
    ]
    good_vg = quality_counts.get("Good", 0) + quality_counts.get("Very good", 0)
    poor_vp = quality_counts.get("Poor", 0) + quality_counts.get("Very poor", 0)
    quality_rows.extend(
        [
            {"quality_rating": "Good or very good", "forms": good_vg, "denominator": total_forms, "percent": round(pct(good_vg, total_forms), 1)},
            {"quality_rating": "Poor or very poor", "forms": poor_vp, "denominator": total_forms, "percent": round(pct(poor_vp, total_forms), 1)},
        ]
    )

    clinician_rows = []
    by_clinician: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in encounters:
        by_clinician[row[args.clinician_id_column]].append(row)
    for clinician_id, rows in sorted(by_clinician.items()):
        forms = len(rows)
        major = sum(yes(row[args.major_error_column]) for row in rows)
        minor = sum(yes(row[args.minor_error_column]) for row in rows)
        specialty = rows[0][args.specialty_column]
        clinician_rows.append(
            {
                "clinician_id": clinician_id,
                "specialty": specialty,
                "forms": forms,
                "major_error_forms": major,
                "minor_error_forms": minor,
                "major_error_rate": safe_rate(major, forms),
                "minor_error_rate": safe_rate(minor, forms),
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "sample_primary_endpoint_summary.csv", summary_rows, args.delimiter)
    write_csv(output_dir / "primary_endpoint_by_specialty.csv", specialty_rows, args.delimiter)
    write_csv(output_dir / "draft_quality_summary.csv", quality_rows, args.delimiter)
    write_csv(output_dir / "figure_clinician_error_counts.csv", clinician_rows, args.delimiter)
    print(f"Wrote aggregate outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
