#!/usr/bin/env python3
"""Questionnaire outcomes: paired workload/time outcomes and acceptance scores."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from publication_utils import bh_adjust, clean, numeric_first_token, write_csv


PANEL_A_ROWS = [
    ("documentation_minutes", "Documentation minutes per consultation"),
    ("consultations_per_day", "Consultations per workday"),
    ("nasa_mental_demand", "NASA-TLX mental demand"),
    ("nasa_time_pressure", "NASA-TLX time pressure"),
    ("nasa_effort", "NASA-TLX effort"),
    ("nasa_frustration", "NASA-TLX frustration"),
    ("nasa_performance", "NASA-TLX perceived task performance"),
]

UTAUT_DOMAIN_SPECS = [
    ("post_utaut_acceptance_index", "Overall UTAUT acceptance index", None),
    ("performance_expectancy", "Performance expectancy / perceived usefulness", list(range(1, 9))),
    ("effort_expectancy", "Effort expectancy / ease of use", list(range(9, 14))),
    ("social_influence", "Social influence", list(range(14, 18))),
    ("facilitating_conditions", "Facilitating conditions / system support", list(range(18, 25))),
    ("behavioral_intention", "Behavioral intention", list(range(25, 28))),
]

UTAUT_THRESHOLD_SPECS = [
    ("post_utaut_item_01", "ADS was useful in practice", 4.0),
    ("post_utaut_item_12", "ADS was easy to use", 4.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-questionnaire", required=True, help="Baseline questionnaire CSV.")
    parser.add_argument("--followup-questionnaire", required=True, help="Follow-up questionnaire CSV.")
    parser.add_argument("--usage-composites", required=True, help="Questionnaire composite-score CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for aggregate outputs.")
    parser.add_argument("--participant-id-column", default="participant_id")
    parser.add_argument("--delimiter", default=";")
    return parser.parse_args()


def first_by_id(df: pd.DataFrame, key_col: str) -> tuple[dict[str, dict[str, object]], int]:
    out: dict[str, dict[str, object]] = {}
    duplicates = 0
    for row in df.to_dict(orient="records"):
        key = clean(row.get(key_col))
        if not key:
            continue
        if key in out:
            duplicates += 1
            continue
        out[key] = row
    return out, duplicates


def mean_ci(diffs: np.ndarray) -> tuple[float, float]:
    if len(diffs) < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return mean, mean
    margin = float(stats.t.ppf(0.975, len(diffs) - 1) * sd / math.sqrt(len(diffs)))
    return mean - margin, mean + margin


def wilcoxon_p(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(stats.wilcoxon(nonzero, zero_method="wilcox", correction=False, alternative="two-sided", method="auto").pvalue)


def fmt_p(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def build_paired_rows(baseline: pd.DataFrame, followup: pd.DataFrame, participant_id_column: str) -> list[dict[str, object]]:
    pre_lookup, pre_duplicates = first_by_id(baseline, participant_id_column)
    post_lookup, post_duplicates = first_by_id(followup, participant_id_column)
    participants = sorted(set(pre_lookup).intersection(post_lookup))
    specs = [
        ("nasa_mental_demand", baseline.columns[5], followup.columns[5]),
        ("nasa_time_pressure", baseline.columns[6], followup.columns[6]),
        ("nasa_effort", baseline.columns[7], followup.columns[7]),
        ("nasa_frustration", baseline.columns[8], followup.columns[8]),
        ("nasa_performance", baseline.columns[9], followup.columns[9]),
        ("documentation_minutes", baseline.columns[28], followup.columns[10]),
        ("consultations_per_day", baseline.columns[29], followup.columns[11]),
    ]

    rows: list[dict[str, object]] = []
    for endpoint_id, pre_col, post_col in specs:
        pre_vals: list[float] = []
        post_vals: list[float] = []
        for participant in participants:
            pre_value = numeric_first_token(pre_lookup[participant].get(pre_col))
            post_value = numeric_first_token(post_lookup[participant].get(post_col))
            if pre_value is None or post_value is None:
                continue
            pre_vals.append(pre_value)
            post_vals.append(post_value)
        diffs = np.array(post_vals, dtype=float) - np.array(pre_vals, dtype=float)
        ci_low, ci_high = mean_ci(diffs)
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "n_paired": len(diffs),
                "baseline_mean": float(np.mean(pre_vals)) if pre_vals else float("nan"),
                "follow_up_mean": float(np.mean(post_vals)) if post_vals else float("nan"),
                "mean_change": float(np.mean(diffs)) if len(diffs) else float("nan"),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wilcoxon_p": wilcoxon_p(diffs),
            }
        )
    adjusted = bh_adjust(row["wilcoxon_p"] for row in rows)
    label_lookup = dict(PANEL_A_ROWS)
    for row, p_bh in zip(rows, adjusted):
        row["adjusted_p_value"] = p_bh
        row["outcome"] = label_lookup[row["endpoint_id"]]
        row["mean_change_95ci"] = f"{row['mean_change']:.2f} ({row['ci95_low']:.2f} to {row['ci95_high']:.2f})"
    rows.append(
        {
            "endpoint_id": "questionnaire_qc",
            "outcome": "Questionnaire linkage QC",
            "n_paired": len(participants),
            "baseline_mean": "",
            "follow_up_mean": "",
            "mean_change": "",
            "ci95_low": "",
            "ci95_high": "",
            "wilcoxon_p": "",
            "adjusted_p_value": "",
            "mean_change_95ci": f"baseline duplicate rows ignored={pre_duplicates}; follow-up duplicate rows ignored={post_duplicates}",
        }
    )
    return rows


def domain_mean(df: pd.DataFrame, item_numbers: list[int]) -> pd.Series:
    cols = [f"post_utaut_item_{number:02d}" for number in item_numbers]
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def mean_sd(series: pd.Series) -> str:
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    return f"{float(clean_series.mean()):.2f} ({float(clean_series.std(ddof=1)):.2f})"


def build_post_rows(usage: pd.DataFrame) -> list[dict[str, object]]:
    if "post_questionnaire_present" in usage.columns:
        usage = usage[pd.to_numeric(usage["post_questionnaire_present"], errors="coerce") == 1].copy()
    rows: list[dict[str, object]] = []
    for key, label, item_numbers in UTAUT_DOMAIN_SPECS:
        series = pd.to_numeric(usage[key], errors="coerce") if item_numbers is None else domain_mean(usage, item_numbers)
        series = series.dropna()
        rows.append(
            {
                "section": "Post-implementation acceptance scores",
                "outcome": label,
                "n": int(series.shape[0]),
                "mean_sd_1_to_5": mean_sd(series),
            }
        )
    for key, label, threshold in UTAUT_THRESHOLD_SPECS:
        series = pd.to_numeric(usage[key], errors="coerce").dropna()
        count = int((series >= threshold).sum())
        denominator = int(series.shape[0])
        percent = round(100.0 * count / denominator, 1) if denominator else 0.0
        rows.append(
            {
                "section": "Post-implementation item-level acceptance thresholds",
                "outcome": label,
                "n": denominator,
                "mean_sd_1_to_5": "",
                "threshold": f">={threshold:.0f}",
                "threshold_count": count,
                "threshold_percent": percent,
                "display": f"{count}/{denominator} ({percent:.1f})",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    baseline = pd.read_csv(args.baseline_questionnaire, sep=args.delimiter, dtype=str).fillna("")
    followup = pd.read_csv(args.followup_questionnaire, sep=args.delimiter, dtype=str).fillna("")
    usage = pd.read_csv(args.usage_composites, sep=args.delimiter)

    paired_rows = build_paired_rows(baseline, followup, args.participant_id_column)
    post_rows = build_post_rows(usage)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "questionnaire_paired_outcomes.csv", paired_rows, args.delimiter)
    write_csv(output_dir / "questionnaire_post_acceptance.csv", post_rows, args.delimiter)
    print(f"Wrote aggregate outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
