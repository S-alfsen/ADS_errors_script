#!/usr/bin/env python3
"""Summarize linked-note text carryover metrics without exporting note text."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from publication_utils import as_float, fmt_float, median, pct, quantile, read_csv, require_columns, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linked-note-pairs", required=True, help="Linked encounter-level note-pair metrics CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for aggregate outputs.")
    parser.add_argument("--clinician-id-column", default="clinician_id")
    parser.add_argument("--specialty-column", default="specialty")
    parser.add_argument("--pair-left-id-column", default="ads_note_id")
    parser.add_argument("--pair-right-id-column", default="ehr_note_id")
    parser.add_argument("--initial-draft-to-ehr-metric", default="initial_draft_to_ehr_reuse_ratio")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--delimiter", default=";")
    return parser.parse_args()


def metric_summary(rows: list[dict[str, str]], metric: str, label: str) -> dict[str, object]:
    values = [value for row in rows if (value := as_float(row.get(metric))) is not None]
    if not values:
        return {"metric": metric, "label": label, "n": 0}
    return {
        "metric": metric,
        "label": label,
        "n": len(values),
        "mean": fmt_float(sum(values) / len(values)),
        "median": fmt_float(median(values)),
        "q1": fmt_float(quantile(values, 0.25)),
        "q3": fmt_float(quantile(values, 0.75)),
        "iqr": fmt_float(quantile(values, 0.75) - quantile(values, 0.25)),
        "min": fmt_float(min(values)),
        "max": fmt_float(max(values)),
        "share_ge_0_60": fmt_float(sum(value >= 0.60 for value in values) / len(values)),
        "n_gt_0_90": sum(value > 0.90 for value in values),
        "share_gt_0_90": fmt_float(sum(value > 0.90 for value in values) / len(values)),
    }


def pair_key(row: dict[str, str], left_col: str, right_col: str) -> tuple[str, str]:
    return ((row.get(left_col) or "").strip(), (row.get(right_col) or "").strip())


def main() -> int:
    args = parse_args()
    rows = read_csv(args.linked_note_pairs, args.delimiter)
    require_columns(
        rows,
        [
            args.clinician_id_column,
            args.specialty_column,
            args.pair_left_id_column,
            args.pair_right_id_column,
            args.initial_draft_to_ehr_metric,
        ],
        "linked-note-pairs",
    )

    overall_rows = [
        metric_summary(
            rows,
            args.initial_draft_to_ehr_metric,
            "Final EHR text traced to the initial ADS draft",
        )
    ]
    population_row = {
        "metric": "analysis_population",
        "label": "Linked note-pair analysis population",
        "n": len(rows),
        "mean": "",
        "median": "",
        "q1": "",
        "q3": "",
        "iqr": "",
        "min": "",
        "max": "",
        "share_ge_0_60": "",
        "clinicians": len({row[args.clinician_id_column] for row in rows}),
    }
    overall_rows.insert(0, population_row)

    threshold_rows = []
    values_by_row = [(row, as_float(row.get(args.initial_draft_to_ehr_metric))) for row in rows]
    valid_rows = [row for row, value in values_by_row if value is not None]
    above_rows = [row for row, value in values_by_row if value is not None and value >= args.threshold]
    valid_pairs = {pair_key(row, args.pair_left_id_column, args.pair_right_id_column) for row in valid_rows}
    above_pairs = {pair_key(row, args.pair_left_id_column, args.pair_right_id_column) for row in above_rows}
    threshold_rows.append(
        {
            "summary_level": "overall",
            "group": "all",
            "metric": args.initial_draft_to_ehr_metric,
            "threshold": args.threshold,
            "threshold_rule": ">=",
            "rows_with_metric": len(valid_rows),
            "rows_above_threshold": len(above_rows),
            "row_percent_above_threshold": round(pct(len(above_rows), len(valid_rows)), 1),
            "unique_pairings_with_metric": len(valid_pairs),
            "unique_pairings_above_threshold": len(above_pairs),
            "unique_pairing_percent_above_threshold": round(pct(len(above_pairs), len(valid_pairs)), 1),
            "clinicians_with_metric": len({row[args.clinician_id_column] for row in valid_rows}),
            "clinicians_with_any_above_threshold": len({row[args.clinician_id_column] for row in above_rows}),
        }
    )

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[row[args.specialty_column]].append(row)
    for group, group_rows in sorted(by_group.items()):
        group_values = [(row, as_float(row.get(args.initial_draft_to_ehr_metric))) for row in group_rows]
        valid = [row for row, value in group_values if value is not None]
        above = [row for row, value in group_values if value is not None and value >= args.threshold]
        threshold_rows.append(
            {
                "summary_level": "specialty",
                "group": group,
                "metric": args.initial_draft_to_ehr_metric,
                "threshold": args.threshold,
                "threshold_rule": ">=",
                "rows_with_metric": len(valid),
                "rows_above_threshold": len(above),
                "row_percent_above_threshold": round(pct(len(above), len(valid)), 1),
                "unique_pairings_with_metric": len({pair_key(row, args.pair_left_id_column, args.pair_right_id_column) for row in valid}),
                "unique_pairings_above_threshold": len({pair_key(row, args.pair_left_id_column, args.pair_right_id_column) for row in above}),
                "unique_pairing_percent_above_threshold": round(
                    pct(
                        len({pair_key(row, args.pair_left_id_column, args.pair_right_id_column) for row in above}),
                        len({pair_key(row, args.pair_left_id_column, args.pair_right_id_column) for row in valid}),
                    ),
                    1,
                ),
                "clinicians_with_metric": len({row[args.clinician_id_column] for row in valid}),
                "clinicians_with_any_above_threshold": len({row[args.clinician_id_column] for row in above}),
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "text_carryover_overall_summary.csv", overall_rows, args.delimiter)
    write_csv(output_dir / "text_carryover_threshold_summary.csv", threshold_rows, args.delimiter)
    print(f"Wrote aggregate outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
