"""Per-stop money model for an end-of-line palletizer against a contracted availability budget.

cases lost per shift = stops per shift x site MTTR (min) x case rate (cases/min)
The rig's 2.9 s auto-resume is NOT a palletizer's recovery time; site MTTR is a swept input.
"""
from __future__ import annotations

MTTR_S = [2.9, 30.0, 120.0, 300.0]          # rig auto-resume, quick reset, typical manual reset, long reset
STOPS_PER_SHIFT = [1, 2, 5, 10, 20]


def budget_min(shift_h: float, availability: float) -> float:
    """Downtime allowed per shift by a contracted availability."""
    return shift_h * 60.0 * (1.0 - availability)


def cases_lost(stops_per_shift: float, mttr_s: float, case_rate_per_min: float) -> float:
    return stops_per_shift * (mttr_s / 60.0) * case_rate_per_min


def max_stops(shift_h: float, availability: float, mttr_s: float) -> float:
    """Most stops per shift that still fit inside the availability budget."""
    return budget_min(shift_h, availability) / (mttr_s / 60.0)


def sensitivity(case_rate: float = 12.0, shift_h: float = 8.0, target_a: float = 0.996,
                mttrs=MTTR_S, stops=STOPS_PER_SHIFT) -> dict:
    rows = []
    for m in mttrs:
        rows.append(dict(mttr_s=m, cases_per_stop=cases_lost(1, m, case_rate),
                         max_stops_in_budget=max_stops(shift_h, target_a, m),
                         cases_lost=[cases_lost(s, m, case_rate) for s in stops],
                         availability=[1 - s * m / 60.0 / (shift_h * 60.0) for s in stops]))
    return dict(case_rate=case_rate, shift_h=shift_h, target_availability=target_a,
                budget_min=budget_min(shift_h, target_a), stops_per_shift=list(stops), rows=rows)
