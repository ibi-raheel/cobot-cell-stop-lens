"""Translate measured availability into cases lost at an end-of-line palletizer."""


def cases_lost(case_rate_per_min: float, shift_h: float, availability: float, caught_fraction: float = 0.0) -> dict:
    shift_min = shift_h * 60.0
    down_min = shift_min * (1.0 - availability)
    # assumption: every caught stop is avoided entirely (upper bound on benefit)
    down_min_after = down_min * (1.0 - caught_fraction)
    return dict(shift_min=shift_min, down_min=down_min, cases_lost=case_rate_per_min * down_min,
                down_min_after=down_min_after, cases_lost_after=case_rate_per_min * down_min_after)
