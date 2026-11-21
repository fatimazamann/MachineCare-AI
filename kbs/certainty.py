"""MYCIN certainty-factor calculus (Shortliffe & Buchanan, 1975).

A certainty factor (CF) lies in [-1, 1]:
    +1  definitely true, 0 unknown, -1 definitely false.
"""

from __future__ import annotations

from typing import Iterable


def clamp(cf: float) -> float:
    """Keep a CF inside [-1, 1] and remove floating-point noise."""
    cf = max(-1.0, min(1.0, float(cf)))
    return round(cf, 6)


def combine(x: float, y: float) -> float:
    """Parallel combination of two CFs for the same hypothesis.

    * both positive:  x + y(1 - x)
    * both negative:  x + y(1 + x)
    * opposite signs: (x + y) / (1 - min(|x|, |y|))
    The function is commutative and associative, so the order in which
    evidence arrives does not change the result.
    """
    x, y = clamp(x), clamp(y)
    if x >= 0 and y >= 0:
        return clamp(x + y * (1 - x))
    if x <= 0 and y <= 0:
        return clamp(x + y * (1 + x))
    denom = 1 - min(abs(x), abs(y))
    if denom == 0:  # +1 combined with -1: total contradiction -> unknown
        return 0.0
    return clamp((x + y) / denom)


def combine_all(cfs: Iterable[float]) -> float:
    result = 0.0
    for cf in cfs:
        result = combine(result, cf)
    return result


def conjunction(cfs: Iterable[float]) -> float:
    """CF of an AND of premises = minimum."""
    values = list(cfs)
    return min(values) if values else 0.0


def disjunction(cfs: Iterable[float]) -> float:
    """CF of an OR of premises = maximum."""
    values = list(cfs)
    return max(values) if values else 0.0


def negation(cf: float) -> float:
    return clamp(-cf)


def propagate(rule_cf: float, premise_cf: float) -> float:
    """Contribution of a fired rule: CF(rule) x max(0, CF(premise))."""
    return clamp(rule_cf * max(0.0, premise_cf))


def label(cf: float, thresholds: dict | None = None) -> str:
    """Human-readable status for a hypothesis CF."""
    t = thresholds or {"confirmed": 0.8, "probable": 0.5, "possible": 0.2}
    if cf >= t["confirmed"]:
        return "Confirmed"
    if cf >= t["probable"]:
        return "Probable"
    if cf >= t["possible"]:
        return "Possible"
    if cf <= -t["probable"]:
        return "Ruled out"
    if cf <= -t["possible"]:
        return "Unlikely"
    return "Unknown"


def verbal(cf: float) -> str:
    """Verbal scale used when eliciting CFs from experts."""
    a = abs(cf)
    if a >= 0.9:
        word = "almost certainly"
    elif a >= 0.7:
        word = "very probably"
    elif a >= 0.5:
        word = "probably"
    elif a >= 0.3:
        word = "possibly"
    elif a > 0:
        word = "weakly suggests"
    else:
        return "unknown"
    return word if cf >= 0 else word + " not"
