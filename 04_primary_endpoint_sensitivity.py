#!/usr/bin/env python3
"""Clinician-cluster sensitivity analyses and deployment time trend."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from publication_utils import fmt_float, read_csv, require_columns, write_csv, yes


QUALITY_SCORE = {
    "Svært dårlig": 1,
    "Dårlig": 2,
    "Verken god eller dårlig": 3,
    "God": 4,
    "Svært god": 5,
    "Very poor": 1,
    "Poor": 2,
    "Neutral": 3,
    "Good": 4,
    "Very good": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encounters", required=True, help="Main-cohort encounter-level CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for aggregate outputs.")
    parser.add_argument("--clinician-id-column", default="clinician_id")
    parser.add_argument("--specialty-column", default="specialty")
    parser.add_argument("--major-error-column", default="major_error_flag")
    parser.add_argument("--minor-error-column", default="minor_error_flag")
    parser.add_argument("--quality-column", default="note_quality")
    parser.add_argument("--timestamp-column", default="form_end")
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--delimiter", default=";")
    return parser.parse_args()


def cluster_sensitivity(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_clinician: dict[str, list[int]] = defaultdict(list)
    specialty_by_clinician: dict[str, str] = {}
    encounter_pairs: list[tuple[str, int]] = []
    for row in rows:
        clinician = row[args.clinician_id_column]
        y = int(yes(row[args.major_error_column]))
        by_clinician[clinician].append(y)
        specialty_by_clinician[clinician] = row[args.specialty_column]
        encounter_pairs.append((clinician, y))

    clinician_rows = []
    for clinician, values in sorted(by_clinician.items()):
        n = len(values)
        major = sum(values)
        clinician_rows.append(
            {
                "clinician_id": clinician,
                "specialty": specialty_by_clinician.get(clinician, ""),
                "forms": n,
                "major_error_forms": major,
                "major_error_rate": fmt_float(major / n if n else None),
            }
        )

    y = np.array([value for _, value in encounter_pairs], dtype=float)
    groups = np.array([clinician for clinician, _ in encounter_pairs])
    n_encounters = len(y)
    n_clinicians = len(by_clinician)
    encounter_rate = float(np.mean(y))
    naive_se = float(np.sqrt(encounter_rate * (1 - encounter_rate) / n_encounters))
    naive_low = encounter_rate - 1.96 * naive_se
    naive_high = encounter_rate + 1.96 * naive_se

    model = sm.OLS(y, np.ones((n_encounters, 1))).fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
    cluster_se = float(model.bse[0])
    cluster_low, cluster_high = [float(value) for value in model.conf_int()[0]]

    clinician_rates = np.array([np.mean(values) for values in by_clinician.values()], dtype=float)
    equal_weight_rate = float(np.mean(clinician_rates))

    rng = np.random.default_rng(args.seed)
    clinician_ids = np.array(list(by_clinician))
    boot_encounter = np.empty(args.bootstrap_reps, dtype=float)
    boot_clinician = np.empty(args.bootstrap_reps, dtype=float)
    for idx in range(args.bootstrap_reps):
        sample_ids = rng.choice(clinician_ids, size=len(clinician_ids), replace=True)
        sampled_rates = []
        sampled_encounters = []
        for clinician_id in sample_ids:
            values = by_clinician[str(clinician_id)]
            sampled_rates.append(float(np.mean(values)))
            sampled_encounters.extend(values)
        boot_encounter[idx] = float(np.mean(sampled_encounters))
        boot_clinician[idx] = float(np.mean(sampled_rates))

    boot_encounter_low, boot_encounter_high = [float(value) for value in np.quantile(boot_encounter, [0.025, 0.975])]
    boot_clinician_low, boot_clinician_high = [float(value) for value in np.quantile(boot_clinician, [0.025, 0.975])]
    design_effect = (cluster_se**2) / (naive_se**2) if naive_se else float("nan")
    effective_n = n_encounters / design_effect if design_effect == design_effect and design_effect else float("nan")

    summary_rows = [
        {
            "estimand": "encounter_level_rate_naive_binomial",
            "n_encounters": n_encounters,
            "n_clinicians": n_clinicians,
            "estimate": fmt_float(encounter_rate),
            "se": fmt_float(naive_se),
            "ci95_low": fmt_float(naive_low),
            "ci95_high": fmt_float(naive_high),
            "notes": "Encounter-level major-error rate with naive binomial variance.",
        },
        {
            "estimand": "encounter_level_rate_cluster_robust",
            "n_encounters": n_encounters,
            "n_clinicians": n_clinicians,
            "estimate": fmt_float(encounter_rate),
            "se": fmt_float(cluster_se),
            "ci95_low": fmt_float(cluster_low),
            "ci95_high": fmt_float(cluster_high),
            "notes": "Encounter-level major-error rate with clinician-cluster robust SE from intercept-only OLS.",
        },
        {
            "estimand": "encounter_level_rate_cluster_bootstrap",
            "n_encounters": n_encounters,
            "n_clinicians": n_clinicians,
            "estimate": fmt_float(encounter_rate),
            "se": fmt_float(float(np.std(boot_encounter, ddof=1))),
            "ci95_low": fmt_float(boot_encounter_low),
            "ci95_high": fmt_float(boot_encounter_high),
            "notes": f"Encounter-level major-error rate with {args.bootstrap_reps} clinician-resampled bootstrap replicates.",
        },
        {
            "estimand": "clinician_equal_weight_rate_bootstrap",
            "n_encounters": n_encounters,
            "n_clinicians": n_clinicians,
            "estimate": fmt_float(equal_weight_rate),
            "se": fmt_float(float(np.std(boot_clinician, ddof=1))),
            "ci95_low": fmt_float(boot_clinician_low),
            "ci95_high": fmt_float(boot_clinician_high),
            "notes": "Sensitivity estimand giving equal weight to each clinician.",
        },
        {
            "estimand": "design_effect_summary",
            "n_encounters": n_encounters,
            "n_clinicians": n_clinicians,
            "estimate": fmt_float(design_effect),
            "se": "",
            "ci95_low": fmt_float(effective_n),
            "ci95_high": "",
            "notes": "Estimate is design effect; ci95_low reports effective sample size.",
        },
    ]
    return clinician_rows, summary_rows


def fit_cluster_trend(df: pd.DataFrame, outcome_col: str) -> tuple[float, float, float, float, float]:
    x = sm.add_constant(df[["day_index"]].astype(float))
    y = df[outcome_col].astype(float)
    fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": df["clinician_id"], "use_correction": True})
    slope = float(fit.params["day_index"])
    se = float(fit.bse["day_index"])
    p_value = float(fit.pvalues["day_index"])
    low, high = [float(value) for value in fit.conf_int().loc["day_index"].tolist()]
    return slope, se, p_value, low, high


def time_trend(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame_rows = []
    for row in rows:
        timestamp = pd.to_datetime(row[args.timestamp_column], errors="coerce", dayfirst=True)
        if pd.isna(timestamp):
            continue
        frame_rows.append(
            {
                "clinician_id": row[args.clinician_id_column],
                "specialty": row[args.specialty_column],
                "timestamp": timestamp,
                "major_yes": int(yes(row[args.major_error_column])),
                "minor_yes": int(yes(row[args.minor_error_column])),
                "any_error_yes": int(yes(row[args.major_error_column]) or yes(row[args.minor_error_column])),
                "quality_score": QUALITY_SCORE.get((row[args.quality_column] or "").strip(), np.nan),
            }
        )
    df = pd.DataFrame(frame_rows).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("No parseable timestamps found")
    start_date = df["timestamp"].min().date()
    df["day_index"] = (df["timestamp"].dt.date - start_date).apply(lambda value: value.days)
    df["study_week"] = (df["day_index"] // 7 + 1).astype(int)

    weekly_rows = []
    for week, part in df.groupby("study_week", sort=True):
        n = len(part)
        weekly_rows.append(
            {
                "study_week": int(week),
                "forms": n,
                "major_error_forms": int(part["major_yes"].sum()),
                "major_error_rate": fmt_float(float(part["major_yes"].mean())),
                "minor_error_forms": int(part["minor_yes"].sum()),
                "minor_error_rate": fmt_float(float(part["minor_yes"].mean())),
                "any_error_forms": int(part["any_error_yes"].sum()),
                "any_error_rate": fmt_float(float(part["any_error_yes"].mean())),
                "mean_quality_score": fmt_float(float(part["quality_score"].dropna().mean())),
            }
        )

    test_rows = []
    for outcome_col, label in [
        ("major_yes", "major_error_rate"),
        ("minor_yes", "minor_error_rate"),
        ("any_error_yes", "any_error_rate"),
        ("quality_score", "quality_score"),
    ]:
        part = df[["clinician_id", "day_index", outcome_col]].dropna().copy()
        slope, se, p_value, low, high = fit_cluster_trend(part, outcome_col)
        test_rows.append(
            {
                "analysis": "cluster_robust_linear_trend",
                "outcome": label,
                "n_forms": len(part),
                "n_clinicians": int(part["clinician_id"].nunique()),
                "slope_per_day": fmt_float(slope),
                "se": fmt_float(se),
                "ci95_low": fmt_float(low),
                "ci95_high": fmt_float(high),
                "p_value": fmt_float(p_value),
                "notes": "Linear probability model with clinician-cluster robust SE; slope is absolute change per day since study start.",
            }
        )
    return weekly_rows, test_rows


def main() -> int:
    args = parse_args()
    rows = read_csv(args.encounters, args.delimiter)
    require_columns(
        rows,
        [
            args.clinician_id_column,
            args.specialty_column,
            args.major_error_column,
            args.minor_error_column,
            args.quality_column,
            args.timestamp_column,
        ],
        "encounters",
    )
    clinician_rows, cluster_rows = cluster_sensitivity(rows, args)
    weekly_rows, trend_rows = time_trend(rows, args)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "primary_endpoint_clinician_summary.csv", clinician_rows, args.delimiter)
    write_csv(output_dir / "primary_endpoint_cluster_sensitivity.csv", cluster_rows, args.delimiter)
    write_csv(output_dir / "primary_endpoint_weekly_summary.csv", weekly_rows, args.delimiter)
    write_csv(output_dir / "primary_endpoint_time_trend_tests.csv", trend_rows, args.delimiter)
    print(f"Wrote aggregate outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
