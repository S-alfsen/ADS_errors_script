#!/usr/bin/env python3
"""Exploratory clinician-level variation, logistic baseline associations, and adherence checks."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, fisher_exact, spearmanr

from publication_utils import (
    as_float,
    fmt_float,
    median,
    normalized_ascii,
    quantile,
    read_csv,
    require_columns,
    safe_rate,
    wilson_ci,
    write_csv,
    yes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encounters", required=True, help="Main-cohort encounter-level CSV.")
    parser.add_argument("--eligible-notes", required=True, help="Clinician-level eligible ADS-note counts CSV.")
    parser.add_argument("--baseline-questionnaire", required=True, help="Baseline questionnaire CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for aggregate outputs.")
    parser.add_argument("--clinician-id-column", default="clinician_id")
    parser.add_argument("--specialty-column", default="specialty")
    parser.add_argument("--major-error-column", default="major_error_flag")
    parser.add_argument("--minor-error-column", default="minor_error_flag")
    parser.add_argument("--eligible-note-count-column", default="eligible_ads_notes")
    parser.add_argument("--participant-id-column", default="participant_id")
    parser.add_argument("--baseline-sex-column", default="sex")
    parser.add_argument("--baseline-age-column", default="age_group")
    parser.add_argument("--baseline-experience-column", default="years_clinical_experience")
    parser.add_argument("--baseline-prior-ads-use-column", default="prior_ads_use")
    parser.add_argument("--baseline-dialect-column", default="dialect_text")
    parser.add_argument("--baseline-speaking-tempo-column", default="speaking_tempo")
    parser.add_argument("--monte-carlo-draws", type=int, default=200000)
    parser.add_argument("--random-seed", type=int, default=20260218)
    parser.add_argument("--delimiter", default=";")
    return parser.parse_args()


def mc_p_value(n: np.ndarray, p: float, observed_x2: float, draws: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    ge_count = 0
    remaining = draws
    chunk = 20000
    while remaining:
        take = min(chunk, remaining)
        simulated = rng.binomial(n.astype(int)[None, :], p, size=(take, len(n)))
        sim_x2 = ((simulated - n * p) ** 2 / (n * p * (1 - p))).sum(axis=1)
        ge_count += int((sim_x2 >= observed_x2).sum())
        remaining -= take
    return (ge_count + 1) / (draws + 1)


def clinician_variation(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_clinician: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_clinician[row[args.clinician_id_column]].append(row)

    summary_rows = []
    for clinician, clinician_rows in sorted(by_clinician.items()):
        n = len(clinician_rows)
        major = sum(yes(row[args.major_error_column]) for row in clinician_rows)
        minor = sum(yes(row[args.minor_error_column]) for row in clinician_rows)
        major_low, major_high = wilson_ci(major, n)
        minor_low, minor_high = wilson_ci(minor, n)
        summary_rows.append(
            {
                "clinician_id": clinician,
                "specialty": clinician_rows[0][args.specialty_column],
                "forms": n,
                "major_error_forms": major,
                "major_error_rate": fmt_float(safe_rate(major, n)),
                "major_rate_ci95_low": fmt_float(major_low),
                "major_rate_ci95_high": fmt_float(major_high),
                "minor_error_forms": minor,
                "minor_error_rate": fmt_float(safe_rate(minor, n)),
                "minor_rate_ci95_low": fmt_float(minor_low),
                "minor_rate_ci95_high": fmt_float(minor_high),
            }
        )

    frame = pd.DataFrame(summary_rows)
    tests = []
    for endpoint, yes_col in [("major", "major_error_forms"), ("minor", "minor_error_forms")]:
        n = frame["forms"].to_numpy(dtype=float)
        k = frame[yes_col].to_numpy(dtype=float)
        pooled = float(k.sum() / n.sum())
        cont = np.column_stack([k, n - k])
        chi2, p_chi, dof, expected = chi2_contingency(cont)
        pearson = float(((k - n * pooled) ** 2 / (n * pooled * (1 - pooled))).sum())
        phi = pearson / (len(frame) - 1)
        mc_p = mc_p_value(n, pooled, pearson, args.monte_carlo_draws, args.random_seed + (1 if endpoint == "minor" else 0))
        tests.append(
            {
                "endpoint": endpoint,
                "clinicians": int(len(frame)),
                "forms": int(n.sum()),
                "overall_rate": fmt_float(pooled),
                "chi_square_stat": fmt_float(float(chi2)),
                "chi_square_df": int(dof),
                "chi_square_p": fmt_float(float(p_chi)),
                "chi_square_expected_min_cell": fmt_float(float(expected.min())),
                "monte_carlo_draws": args.monte_carlo_draws,
                "monte_carlo_p": fmt_float(mc_p),
                "overdispersion_phi": fmt_float(phi),
                "evidence_of_between_clinician_variation": "yes" if mc_p < 0.05 else "no",
            }
        )
    return summary_rows, tests


def age_order(value: object) -> int | None:
    text = normalized_ascii(value)
    if text.startswith("under 30"):
        return 1
    if text.startswith("30"):
        return 2
    if text.startswith("40"):
        return 3
    if text.startswith("50"):
        return 4
    if text.startswith("60"):
        return 5
    return None


def text_key(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    for old, new in {"æ": "ae", "ø": "o", "å": "a", "Æ": "ae", "Ø": "o", "Å": "a"}.items():
        text = text.replace(old, new)
    return normalized_ascii(text)


def recode_dialect(value: object) -> str:
    text = text_key(value)
    if not text or text == "-":
        return "Unclear / missing"
    if "nederlandsk" in text or "svensk opprinnelig" in text or "dutch" in text:
        return "Non-native / non-Norwegian"
    if any(token in text for token in ["sorland", "kristiansand", "grimstad", "lillesand", "mandal", "lokal", "local"]):
        return "Local dialect"
    if any(token in text for token in ["ostland", "bokmal", "oslo", "east"]):
        return "Eastern / written-standard dialect"
    if any(token in text for token in ["rogaland", "molde", "tronder", "bergensk", "bergen"]):
        return "Other Norwegian dialect"
    return "Other Norwegian dialect"


def recode_tempo(value: object) -> str:
    text = text_key(value)
    if "rolig" in text or "calm" in text:
        return "Calm"
    if "middels" in text or "medium" in text:
        return "Medium"
    if "hoyt" in text or "high" in text:
        return "High"
    return "Other / missing"


def pct_display(count: int, denominator: int) -> str:
    return f"{count}/{denominator} ({100 * count / denominator:.1f}%)" if denominator else "0/0"


def p_display(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def or_ci_display(odds_ratio: float | None, low: float | None, high: float | None) -> str:
    if odds_ratio is None or low is None or high is None:
        return ""
    if not (math.isfinite(odds_ratio) and math.isfinite(low) and math.isfinite(high)):
        return ""
    return f"{odds_ratio:.2f} ({low:.2f}-{high:.2f})"


def sex_category(value: object) -> str:
    text = normalized_ascii(value)
    if text.startswith("kvinne") or text.startswith("female"):
        return "Female"
    if text.startswith("mann") or text.startswith("male"):
        return "Male"
    return "Missing / other"


def prior_ads_category(value: object) -> str:
    text = normalized_ascii(value)
    if text.startswith("ja") or text.startswith("yes"):
        return "Yes"
    if text.startswith("nei") or text.startswith("no"):
        return "No"
    return "Missing"


def sex_numeric(value: object) -> float | None:
    category = sex_category(value)
    if category == "Female":
        return 1.0
    if category == "Male":
        return 0.0
    return None


def prior_ads_numeric(value: object) -> float | None:
    category = prior_ads_category(value)
    if category == "Yes":
        return 1.0
    if category == "No":
        return 0.0
    return None


def age_group_label(value: object) -> str:
    text = normalized_ascii(value)
    if text.startswith("under 30"):
        return "Under 30"
    if text.startswith("30"):
        return "30-39"
    if text.startswith("40"):
        return "40-49"
    if text.startswith("50"):
        return "50-59"
    if text.startswith("60"):
        return "60+"
    return "Missing"


def baseline_form_level_frame(rows: list[dict[str, str]], args: argparse.Namespace) -> pd.DataFrame:
    baseline = pd.read_csv(args.baseline_questionnaire, sep=args.delimiter, dtype=str).fillna("")

    baseline_rows = []
    for row in baseline.to_dict(orient="records"):
        clinician = (row.get(args.participant_id_column) or "").strip()
        if not clinician:
            continue
        dialect = recode_dialect(row.get(args.baseline_dialect_column, ""))
        tempo = recode_tempo(row.get(args.baseline_speaking_tempo_column, ""))
        baseline_rows.append(
            {
                "clinician_id": clinician,
                "sex_category": sex_category(row.get(args.baseline_sex_column, "")),
                "sex_female": sex_numeric(row.get(args.baseline_sex_column, "")),
                "age_group": age_group_label(row.get(args.baseline_age_column, "")),
                "age_group_order": age_order(row.get(args.baseline_age_column, "")),
                "years_clinical_experience": as_float(row.get(args.baseline_experience_column, "")),
                "prior_ads_use_category": prior_ads_category(row.get(args.baseline_prior_ads_use_column, "")),
                "prior_ads_use_yes": prior_ads_numeric(row.get(args.baseline_prior_ads_use_column, "")),
                "speaking_tempo_category": tempo,
                "speaking_tempo_order": {"Calm": 1, "Medium": 2, "High": 3}.get(tempo),
                "dialect_category": dialect,
                "local_dialect": 1.0 if dialect == "Local dialect" else 0.0,
                "non_native_dialect": 1.0 if dialect == "Non-native / non-Norwegian" else 0.0,
            }
        )

    baseline_frame = pd.DataFrame(baseline_rows)
    form_rows = []
    for row in rows:
        form_rows.append(
            {
                "clinician_id": (row.get(args.clinician_id_column) or "").strip(),
                "major_error": int(yes(row.get(args.major_error_column, ""))),
                "any_error": int(yes(row.get(args.major_error_column, "")) or yes(row.get(args.minor_error_column, ""))),
            }
        )
    form_frame = pd.DataFrame(form_rows)
    frame = form_frame.merge(baseline_frame, on="clinician_id", how="inner")
    frame["years_clinical_experience_10"] = frame["years_clinical_experience"] / 10.0
    return frame


def observed_group_rates(frame: pd.DataFrame, outcome: str, column: str, labels: list[str]) -> str:
    parts = []
    for label in labels:
        part = frame[frame[column] == label]
        if part.empty:
            continue
        parts.append(f"{label}: {pct_display(int(part[outcome].sum()), len(part))}")
    return "; ".join(parts)


def continuous_summary(frame: pd.DataFrame, column: str, units: str) -> str:
    clinician_values = frame[["clinician_id", column]].drop_duplicates().dropna()[column].astype(float).tolist()
    if not clinician_values:
        return ""
    return (
        f"Clinician median {median(clinician_values):.1f} {units} "
        f"(IQR {quantile(clinician_values, 0.25):.1f}-{quantile(clinician_values, 0.75):.1f})"
    )


def binary_observed_rates(frame: pd.DataFrame, outcome: str, column: str, positive_label: str, negative_label: str) -> str:
    parts = []
    for value, label in [(0.0, negative_label), (1.0, positive_label)]:
        part = frame[frame[column] == value]
        if part.empty:
            continue
        parts.append(f"{label}: {pct_display(int(part[outcome].sum()), len(part))}")
    return "; ".join(parts)


def fit_logistic_model(frame: pd.DataFrame, outcome: str, predictor: str) -> dict[str, object]:
    model_frame = frame[["clinician_id", outcome, predictor]].dropna().copy()
    model_frame[predictor] = model_frame[predictor].astype(float)
    if model_frame[predictor].nunique() < 2:
        return {
            "n_clinicians": int(model_frame["clinician_id"].nunique()),
            "n_forms": int(len(model_frame)),
            "outcome_forms": int(model_frame[outcome].sum()),
            "odds_ratio": None,
            "ci95_low": None,
            "ci95_high": None,
            "p_value": None,
        }
    x = sm.add_constant(model_frame[[predictor]], has_constant="add")
    y = model_frame[outcome].astype(float)
    result = sm.GLM(y, x, family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_frame["clinician_id"], "use_correction": True},
    )
    coefficient = float(result.params[predictor])
    ci_low, ci_high = [float(value) for value in result.conf_int().loc[predictor].tolist()]
    return {
        "n_clinicians": int(model_frame["clinician_id"].nunique()),
        "n_forms": int(len(model_frame)),
        "outcome_forms": int(model_frame[outcome].sum()),
        "odds_ratio": math.exp(coefficient),
        "ci95_low": math.exp(ci_low),
        "ci95_high": math.exp(ci_high),
        "p_value": float(result.pvalues[predictor]),
    }


def baseline_logistic_regression(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = baseline_form_level_frame(rows, args)
    outcome_specs = [
        {
            "outcome_id": "major_error",
            "outcome_label": "Major error",
            "raw_outcome": "major_error_flag",
        },
        {
            "outcome_id": "any_error",
            "outcome_label": "Any error (major or minor)",
            "raw_outcome": "any_error_flag",
        },
    ]
    model_specs = [
        {
            "predictor_id": "sex_female",
            "characteristic": "Sex",
            "comparison": "Female vs male",
            "observed": lambda data, outcome: observed_group_rates(data, outcome, "sex_category", ["Male", "Female"]),
        },
        {
            "predictor_id": "age_group_order",
            "characteristic": "Age group",
            "comparison": "Per one older age category",
            "observed": lambda data, outcome: observed_group_rates(data, outcome, "age_group", ["Under 30", "30-39", "40-49", "50-59", "60+"]),
        },
        {
            "predictor_id": "years_clinical_experience_10",
            "characteristic": "Clinical experience",
            "comparison": "Per 10-year increase",
            "observed": lambda data, outcome: continuous_summary(data, "years_clinical_experience", "years"),
        },
        {
            "predictor_id": "prior_ads_use_yes",
            "characteristic": "Prior ADS use",
            "comparison": "Yes vs no",
            "observed": lambda data, outcome: observed_group_rates(data, outcome, "prior_ads_use_category", ["No", "Yes"]),
        },
        {
            "predictor_id": "speaking_tempo_order",
            "characteristic": "Speaking tempo",
            "comparison": "Per faster category",
            "observed": lambda data, outcome: observed_group_rates(data, outcome, "speaking_tempo_category", ["Calm", "Medium", "High"]),
        },
        {
            "predictor_id": "local_dialect",
            "characteristic": "Reported local dialect",
            "comparison": "Local dialect compared with all other or unclear responses",
            "observed": lambda data, outcome: binary_observed_rates(data, outcome, "local_dialect", "Local dialect", "All other or unclear responses"),
        },
        {
            "predictor_id": "non_native_dialect",
            "characteristic": "Reported non-Norwegian dialect or accent",
            "comparison": "Non-Norwegian dialect or accent compared with all other or unclear responses",
            "observed": lambda data, outcome: binary_observed_rates(data, outcome, "non_native_dialect", "Non-Norwegian dialect or accent", "All other or unclear responses"),
        },
    ]

    raw_rows = []
    table_rows = []
    for outcome_spec in outcome_specs:
        outcome = outcome_spec["outcome_id"]
        for spec in model_specs:
            predictor = spec["predictor_id"]
            fit = fit_logistic_model(frame, outcome, predictor)
            odds_ratio = fit["odds_ratio"]
            ci_low = fit["ci95_low"]
            ci_high = fit["ci95_high"]
            p_value = fit["p_value"]
            raw_row = {
                "outcome": outcome_spec["raw_outcome"],
                "outcome_label": outcome_spec["outcome_label"],
                "predictor_id": predictor,
                "clinician_characteristic": spec["characteristic"],
                "modelled_comparison": spec["comparison"],
                "observed_data": spec["observed"](frame.dropna(subset=[predictor]), outcome),
                "n_clinicians": fit["n_clinicians"],
                "n_forms": fit["n_forms"],
                "outcome_forms": fit["outcome_forms"],
                "odds_ratio": fmt_float(odds_ratio, 8),
                "ci95_low": fmt_float(ci_low, 8),
                "ci95_high": fmt_float(ci_high, 8),
                "p_value": fmt_float(p_value, 8),
                "model": "Univariable logistic regression with clinician-cluster-robust standard errors.",
            }
            raw_rows.append(raw_row)
            table_rows.append(
                {
                    "Outcome": outcome_spec["outcome_label"],
                    "Clinician characteristic": spec["characteristic"],
                    "Modelled comparison": spec["comparison"],
                    "Observed data": raw_row["observed_data"],
                    "Clinicians, n": fit["n_clinicians"],
                    "Submitted forms, n": fit["n_forms"],
                    "Outcome forms, n": fit["outcome_forms"],
                    "Odds ratio (95% CI)": or_ci_display(odds_ratio, ci_low, ci_high),
                    "P value": p_display(p_value),
                }
            )
    return raw_rows, table_rows


LOGISTIC_TABLE_NOTES = [
    "Odds ratios are from separate univariable logistic regression models with submitted-form error status as the binary outcome and clinician-cluster-robust standard errors.",
    "Any error was defined as at least one major or minor error on a submitted form.",
    "Age group was modelled as an ordinal predictor from Under 30 to 60+; odds ratio below 1 therefore indicates lower odds in older age categories.",
    "Dialect or accent groupings were based on clinicians' free-text descriptions and should be treated as exploratory. The reported non-Norwegian dialect or accent group comprised 2 clinicians and 26 submitted forms.",
    "Clinician characteristics were available for 24 of 28 clinicians; 4 clinicians contributing 20 submitted forms were not included in these baseline-characteristic models.",
]


def write_markdown_table(path: Path, rows: list[dict[str, object]], notes: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"Note: {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def adherence_bias(rows: list[dict[str, str]], eligible_rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    by_clinician: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_clinician[row[args.clinician_id_column]].append(row)
    eligible_by = {row[args.clinician_id_column]: int(float(row[args.eligible_note_count_column] or 0)) for row in eligible_rows}
    user_rows = []
    for clinician, clinician_rows in sorted(by_clinician.items()):
        notes = eligible_by.get(clinician, 0)
        forms = len(clinician_rows)
        if not forms or not notes:
            continue
        major = sum(yes(row[args.major_error_column]) for row in clinician_rows)
        minor = sum(yes(row[args.minor_error_column]) for row in clinician_rows)
        any_error = sum(yes(row[args.major_error_column]) or yes(row[args.minor_error_column]) for row in clinician_rows)
        user_rows.append(
            {
                "clinician_id": clinician,
                "specialty": clinician_rows[0][args.specialty_column],
                "forms": forms,
                "eligible_ads_notes": notes,
                "submission_adherence": forms / notes,
                "major_error_forms": major,
                "major_error_rate": major / forms,
                "minor_error_forms": minor,
                "minor_error_rate": minor / forms,
                "any_error_forms": any_error,
                "any_error_rate": any_error / forms,
            }
        )

    tests = []
    adherence = [row["submission_adherence"] for row in user_rows]
    for outcome in ["major_error_rate", "minor_error_rate", "any_error_rate"]:
        result = spearmanr(adherence, [row[outcome] for row in user_rows])
        tests.append({"analysis": "spearman", "comparison": f"submission_adherence_vs_{outcome}", "subset": "all_clinicians_with_forms", "clinicians": len(user_rows), "statistic": fmt_float(float(result.statistic)), "p_value": fmt_float(float(result.pvalue))})

    median_value = median(adherence)
    split_defs = {
        "adherence_lt_0.5_vs_ge_0.5": lambda row: row["submission_adherence"] < 0.5,
        "adherence_le_median_vs_gt_median": lambda row: row["submission_adherence"] <= median_value,
    }
    summary_rows = []
    for split_label, is_low in split_defs.items():
        low = [row for row in user_rows if is_low(row)]
        high = [row for row in user_rows if not is_low(row)]
        for label, split_rows in [("lower_adherence", low), ("higher_adherence", high)]:
            forms = sum(row["forms"] for row in split_rows)
            major = sum(row["major_error_forms"] for row in split_rows)
            minor = sum(row["minor_error_forms"] for row in split_rows)
            any_error = sum(row["any_error_forms"] for row in split_rows)
            summary_rows.append({"split": split_label, "subset": label, "clinicians": len(split_rows), "forms": forms, "major_error_forms": major, "major_error_rate": fmt_float(safe_rate(major, forms)), "minor_error_forms": minor, "minor_error_rate": fmt_float(safe_rate(minor, forms)), "any_error_forms": any_error, "any_error_rate": fmt_float(safe_rate(any_error, forms))})
        for outcome in ["major_error_forms", "minor_error_forms", "any_error_forms"]:
            a = sum(row[outcome] for row in low)
            b = sum(row["forms"] - row[outcome] for row in low)
            c = sum(row[outcome] for row in high)
            d = sum(row["forms"] - row[outcome] for row in high)
            result = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            tests.append({"analysis": "fisher_exact", "comparison": outcome, "subset": split_label, "clinicians": len(low) + len(high), "statistic": fmt_float(float(result.statistic)), "p_value": fmt_float(float(result.pvalue))})
    return user_rows, tests, summary_rows


def main() -> int:
    args = parse_args()
    encounters = read_csv(args.encounters, args.delimiter)
    eligible = read_csv(args.eligible_notes, args.delimiter)
    require_columns(encounters, [args.clinician_id_column, args.specialty_column, args.major_error_column, args.minor_error_column], "encounters")
    require_columns(eligible, [args.clinician_id_column, args.eligible_note_count_column], "eligible-notes")

    variation_summary, variation_tests = clinician_variation(encounters, args)
    logistic_models, logistic_table = baseline_logistic_regression(encounters, args)
    adherence_users, adherence_tests, adherence_summary = adherence_bias(encounters, eligible, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_output in ["baseline_association_user_level.csv", "baseline_association_tests.csv"]:
        (output_dir / stale_output).unlink(missing_ok=True)

    write_csv(output_dir / "clinician_error_variation_summary.csv", variation_summary, args.delimiter)
    write_csv(output_dir / "clinician_error_variation_tests.csv", variation_tests, args.delimiter)
    write_csv(output_dir / "baseline_logistic_regression_models.csv", logistic_models, args.delimiter)
    write_csv(output_dir / "baseline_logistic_regression_table.csv", logistic_table, args.delimiter)
    write_markdown_table(output_dir / "baseline_logistic_regression_table_preview.md", logistic_table, LOGISTIC_TABLE_NOTES)
    write_csv(output_dir / "form_adherence_user_level.csv", adherence_users, args.delimiter)
    write_csv(output_dir / "form_adherence_bias_tests.csv", adherence_tests, args.delimiter)
    write_csv(output_dir / "form_adherence_bias_summary.csv", adherence_summary, args.delimiter)
    print(f"Wrote aggregate outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
