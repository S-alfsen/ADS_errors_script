#!/usr/bin/env python3
"""Baseline clinician characteristics (manuscript Table 1)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from publication_utils import (
    as_float,
    count_pct,
    fmt_float,
    median,
    normalized_ascii,
    pct,
    quantile,
    read_csv,
    require_columns,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-questionnaire", required=True, help="Baseline questionnaire CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for aggregate outputs.")
    parser.add_argument("--profession-column", default="profession")
    parser.add_argument("--department-column", default="department")
    parser.add_argument("--sex-column", default="sex")
    parser.add_argument("--age-column", default="age_group")
    parser.add_argument("--experience-column", default="years_clinical_experience")
    parser.add_argument("--language-column", default="clinical_language")
    parser.add_argument("--dialect-column", default="dialect_text")
    parser.add_argument("--delimiter", default=";")
    return parser.parse_args()


def text_key(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    for old, new in {"æ": "ae", "ø": "o", "å": "a"}.items():
        text = text.replace(old, new)
    return normalized_ascii(text)


def recode_profession(value: object) -> str:
    text = text_key(value)
    if not text:
        return "Missing"
    if "psykolog" in text or "psycholog" in text:
        return "Psychologist"
    if "sykepleier" in text or "nurse" in text:
        return "Nurse"
    if "lege" in text or "physician" in text or "doctor" in text:
        return "Physician"
    return "Other"


def recode_department(value: object) -> str:
    text = text_key(value)
    if not text:
        return "Missing"
    if "psyk" in text:
        return "Psychiatric services"
    if "onkolog" in text or "oncolog" in text or "kreft" in text:
        return "Oncology"
    if "ortop" in text or "orthop" in text:
        return "Orthopedics"
    if "otolaryng" in text or "ore" in text or "nese" in text or "hals" in text or "onh" in text:
        return "Otolaryngology"
    return "Other / unrecoded"


def recode_sex(value: object) -> str:
    text = text_key(value)
    if text.startswith(("kvinne", "female", "woman")):
        return "Female"
    if text.startswith(("mann", "male", "man")):
        return "Male"
    return "Missing"


def recode_age(value: object) -> str:
    text = text_key(value)
    if text.startswith("under 30"):
        return "Under 30 years"
    if text.startswith("30"):
        return "30-39 years"
    if text.startswith("40"):
        return "40-49 years"
    if text.startswith("50"):
        return "50-59 years"
    if text.startswith("60"):
        return "60 years or older"
    return "Missing"


def recode_language(value: object) -> str:
    text = text_key(value)
    if not text:
        return "Missing"
    if "svensk" in text or "swedish" in text:
        return "Swedish"
    return "Norwegian or primarily Norwegian"


def recode_dialect(value: object) -> str:
    # Same recoding logic as 05_clinician_exploratory_associations.py,
    # reported with the manuscript Table 1 labels.
    text = text_key(value)
    if not text or text == "-":
        return "Other"
    if "nederlandsk" in text or "svensk opprinnelig" in text or "dutch" in text:
        return "Non-native / non-Norwegian"
    if any(token in text for token in ["sorland", "kristiansand", "grimstad", "lillesand", "mandal", "lokal", "local"]):
        return "Local"
    if any(token in text for token in ["ostland", "bokmal", "oslo", "east"]):
        return "National"
    return "Other Norwegian dialect"


SECTIONS = [
    ("Professional characteristics", "profession", recode_profession, ["Physician", "Psychologist", "Nurse", "Other", "Missing"]),
    ("Department", "department", recode_department, ["Psychiatric services", "Oncology", "Orthopedics", "Otolaryngology", "Other / unrecoded", "Missing"]),
    ("Sex", "sex", recode_sex, ["Female", "Male", "Missing"]),
    ("Age", "age", recode_age, ["Under 30 years", "30-39 years", "40-49 years", "50-59 years", "60 years or older", "Missing"]),
    ("Language", "language", recode_language, ["Norwegian or primarily Norwegian", "Swedish", "Missing"]),
    ("Dialect category", "dialect", recode_dialect, ["Local", "National", "Other Norwegian dialect", "Non-native / non-Norwegian", "Other", "Missing"]),
]


def main() -> int:
    args = parse_args()
    rows = read_csv(args.baseline_questionnaire, args.delimiter)
    columns = {
        "profession": args.profession_column,
        "department": args.department_column,
        "sex": args.sex_column,
        "age": args.age_column,
        "language": args.language_column,
        "dialect": args.dialect_column,
    }
    require_columns(rows, list(columns.values()) + [args.experience_column], "baseline-questionnaire")
    denominator = len(rows)

    output_rows: list[dict[str, object]] = []
    for section, key, recode, levels in SECTIONS:
        counts = Counter(recode(row.get(columns[key], "")) for row in rows)
        unexpected = sorted(set(counts) - set(levels))
        for level in levels + unexpected:
            if not counts.get(level, 0):
                continue
            output_rows.append(
                {
                    "section": section,
                    "level": level,
                    "count": counts[level],
                    "denominator": denominator,
                    "percent": round(pct(counts[level], denominator), 1),
                    "display": count_pct(counts[level], denominator),
                }
            )

    experience = [value for row in rows if (value := as_float(row.get(args.experience_column, ""))) is not None]
    if experience:
        output_rows.append(
            {
                "section": "Clinical experience",
                "level": "Years of clinical experience, median (IQR)",
                "count": len(experience),
                "denominator": denominator,
                "percent": "",
                "display": f"{fmt_float(median(experience))} ({fmt_float(quantile(experience, 0.25))}-{fmt_float(quantile(experience, 0.75))})",
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "baseline_characteristics_summary.csv", output_rows, args.delimiter)
    print(f"Wrote aggregate outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
