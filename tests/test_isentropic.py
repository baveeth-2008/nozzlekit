"""Tests for nozzlekit.isentropic.

Reference values are the standard gamma = 1.4 isentropic flow table
(NACA Report 1135 / Anderson, *Modern Compressible Flow*), rounded to the
number of digits printed in those tables.
"""

import pytest

from nozzlekit import (
    area_ratio,
    density_ratio,
    mach_from_area_ratio,
    pressure_ratio,
    temperature_ratio,
)

# (Mach, A/A*, p/p0, T/T0, rho/rho0) for gamma = 1.4
TABLE = [
    (0.5, 1.3398, 0.8430, 0.9524, 0.8852),
    (1.0, 1.0000, 0.5283, 0.8333, 0.6339),
    (2.0, 1.6875, 0.1278, 0.5556, 0.2300),
    (3.0, 4.2346, 0.02722, 0.3571, 0.07623),
]


@pytest.mark.parametrize("mach, a_ratio, p_ratio, t_ratio, rho_ratio", TABLE)
def test_matches_textbook_table(mach, a_ratio, p_ratio, t_ratio, rho_ratio):
    assert area_ratio(mach, 1.4) == pytest.approx(a_ratio, rel=1e-3)
    assert pressure_ratio(mach, 1.4) == pytest.approx(p_ratio, rel=1e-3)
    assert temperature_ratio(mach, 1.4) == pytest.approx(t_ratio, rel=1e-3)
    assert density_ratio(mach, 1.4) == pytest.approx(rho_ratio, rel=1e-3)


def test_area_ratio_is_one_at_mach_one_for_any_gamma():
    for gamma in (1.1, 1.2, 1.3, 1.4, 1.67):
        assert area_ratio(1.0, gamma) == pytest.approx(1.0)


def test_stagnation_conditions_at_mach_zero():
    assert pressure_ratio(0.0) == 1.0
    assert temperature_ratio(0.0) == 1.0
    assert density_ratio(0.0) == 1.0


def test_equation_of_state_consistency():
    # p / p0 = (rho / rho0) * (T / T0) for an ideal gas
    for mach in (0.3, 1.0, 2.5):
        for gamma in (1.2, 1.4):
            lhs = pressure_ratio(mach, gamma)
            rhs = density_ratio(mach, gamma) * temperature_ratio(mach, gamma)
            assert lhs == pytest.approx(rhs)


def test_supersonic_branch_from_area_ratio():
    # Standard table: A/A* = 2.0 gives M = 2.197 (supersonic), 0.306 (subsonic)
    assert mach_from_area_ratio(2.0, 1.4, supersonic=True) == pytest.approx(2.197, abs=1e-3)
    assert mach_from_area_ratio(2.0, 1.4, supersonic=False) == pytest.approx(0.306, abs=1e-3)


@pytest.mark.parametrize("gamma", [1.15, 1.2, 1.3, 1.4])
@pytest.mark.parametrize("mach", [1.2, 2.0, 3.5, 5.0])
def test_round_trip_supersonic(mach, gamma):
    ratio = area_ratio(mach, gamma)
    assert mach_from_area_ratio(ratio, gamma, supersonic=True) == pytest.approx(mach, rel=1e-9)


@pytest.mark.parametrize("mach", [0.1, 0.3, 0.7, 0.95])
def test_round_trip_subsonic(mach):
    ratio = area_ratio(mach, 1.3)
    assert mach_from_area_ratio(ratio, 1.3, supersonic=False) == pytest.approx(mach, rel=1e-9)


def test_area_ratio_of_exactly_one_gives_mach_one():
    assert mach_from_area_ratio(1.0) == 1.0


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        area_ratio(2.0, gamma=1.0)
    with pytest.raises(ValueError):
        pressure_ratio(-1.0)
    with pytest.raises(ValueError):
        area_ratio(0.0)
    with pytest.raises(ValueError):
        mach_from_area_ratio(0.5)
