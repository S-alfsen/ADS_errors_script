#!/usr/bin/env python3
"""Shared utilities for the public analysis scripts."""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence


def read_csv(path: str | Path, delimiter: str = ";") -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def write_csv(path: str | Path, rows: Sequence[dict[str, object]], delimiter: str = ";") -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def require_columns(rows: Sequence[dict[str, str]], columns: Iterable[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} has no rows")
    observed = set(rows[0])
    missing = [column for column in columns if column not in observed]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def as_float(value: object) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def as_int(value: object) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"Expected integer-like value, got {value!r}")
    return int(number)


def clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def yes(value: object) -> bool:
    return clean(value).lower() in {"ja", "yes", "true", "1"}


def fmt_float(value: float | int | None, digits: int = 12) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def pct(count: int, denominator: int, digits: int = 1) -> float:
    return round(100.0 * count / denominator, digits) if denominator else 0.0


def count_pct(count: int, denominator: int, digits: int = 1) -> str:
    return f"{count}/{denominator} ({pct(count, denominator, digits):.{digits}f})"


def wilson_ci(count: int, denominator: int, confidence: float = 0.95) -> tuple[float, float]:
    if denominator == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054 if confidence == 0.95 else normal_quantile(0.5 + confidence / 2)
    p_hat = count / denominator
    denom = 1 + z * z / denominator
    center = (p_hat + z * z / (2 * denominator)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) / denominator) + z * z / (4 * denominator * denominator)) / denom
    return center - margin, center + margin


def normal_quantile(p: float) -> float:
    # Acklam inverse-normal approximation. Kept here to avoid a scipy dependency
    # for scripts that only need Wilson intervals.
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def bh_adjust(pvalues: Iterable[float | None]) -> list[float | None]:
    values = [None if p is None or math.isnan(p) else float(p) for p in pvalues]
    valid = [(idx, p) for idx, p in enumerate(values) if p is not None]
    adjusted: list[float | None] = [None] * len(values)
    m = len(valid)
    previous = 1.0
    for rank, (idx, p) in reversed(list(enumerate(sorted(valid, key=lambda item: item[1]), start=1))):
        value = min(previous, p * m / rank)
        adjusted[idx] = max(0.0, min(1.0, value))
        previous = value
    return adjusted


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("No values supplied")
    vals = sorted(values)
    idx = (len(vals) - 1) * probability
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return float(vals[low])
    return float(vals[low] + (vals[high] - vals[low]) * (idx - low))


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("No values supplied")
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2)


def numeric_first_token(value: object) -> float | None:
    text = clean(value)
    match = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    return as_float(match.group(0))


def normalized_ascii(value: object) -> str:
    text = clean(value)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii").lower().strip()


def safe_rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None
