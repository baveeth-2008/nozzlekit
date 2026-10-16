"""Tests for nozzlekit.sizing.

Hand checks use gamma = 1.4 so the isentropic exponents are simple fractions
(for example (gamma + 1) / (2 (gamma - 1)) = 3).  Consistency checks use the
identities F = C_F p_c A* = m_dot C_F c* = m_dot I_sp g_0.
"""

import pytest

from nozzlekit.isentropic import area_ratio, mach_from_area_ratio, pressure_ratio
from nozzlekit.sizing import (
    G0,
    R_UNIVERSAL,
    characteristic_velocity,
    exit_area,
    mass_flow_rate,
    specific_impulse,
    throat_area,
    thrust_coefficient,
)


def test_characteristic_velocity_gamma_14_hand_check():
    # gamma = 1.4, T_c = 2000 K, M = 0.02 kg/mol
    # R = 8.314462618 / 0.02
    # (2/(gamma+1))^((gamma+1)/(2(gamma-1))) = (5/6)^3 = 125/216
    # c* = sqrt(gamma R T_c) / (gamma * 125/216)
    gamma = 1.4
    temperature = 2000.0
    molar_mass = 0.02
    gas_constant = R_UNIVERSAL / molar_mass
    sonic_factor = (5.0 / 6.0) ** 3
    expected = (gamma * gas_constant * temperature) ** 0.5 / (gamma * sonic_factor)
    assert characteristic_velocity(temperature, gamma, molar_mass) == pytest.approx(
        expected
    )


def test_thrust_coefficient_sonic_optimum_hand_check():
    # Sonic exit, gamma = 1.4: A_e/A* = 1, p_e/p_c = (5/6)^{3.5}
    # (p_e/p_c)^((gamma-1)/gamma) = 2/(gamma+1) = 5/6
    # C_F,opt = sqrt( (2 gamma²/(gamma-1)) * (5/6)^6 * (1/6) )
    #         = sqrt(9.8 * 15625 / 279936)
    p0 = 1.0e6
    pe = p0 * (5.0 / 6.0) ** 3.5
    expected = (9.8 * 15625.0 / 279936.0) ** 0.5
    cf = thrust_coefficient(p0, pe, 1.4, exit_pressure=pe)
    assert cf == pytest.approx(expected)
    assert cf == pytest.approx(thrust_coefficient(p0, pe, 1.4, area_ratio=1.0))
    # Independent check: at optimum expansion C_F = v_e / c*
    # v_e = M_e sqrt(gamma R T_e), T_e / T_c = 2/(gamma+1) = 5/6 at M_e = 1
    gas_constant = R_UNIVERSAL / 0.02
    t0 = 2000.0
    ve = 1.0 * (1.4 * gas_constant * t0 * 5.0 / 6.0) ** 0.5
    cstar = characteristic_velocity(t0, 1.4, 0.02)
    assert cf == pytest.approx(ve / cstar)


def test_thrust_coefficient_pressure_term_vacuum_vs_optimum():
    # C_F,vac = C_F,opt + (p_e / p_c) * (A_e / A*)
    p0 = 5.0e6
    gamma = 1.4
    mach_exit = 2.0
    expansion = area_ratio(mach_exit, gamma)
    pe = p0 * pressure_ratio(mach_exit, gamma)
    cf_opt = thrust_coefficient(p0, pe, gamma, area_ratio=expansion)
    cf_vac = thrust_coefficient(p0, 0.0, gamma, area_ratio=expansion)
    assert cf_vac == pytest.approx(cf_opt + (pe / p0) * expansion)
    # Overexpanded: ambient above exit pressure reduces C_F
    cf_over = thrust_coefficient(p0, 2.0 * pe, gamma, area_ratio=expansion)
    assert cf_over == pytest.approx(cf_opt - (pe / p0) * expansion)


def test_exit_pressure_and_area_ratio_give_the_same_cf():
    p0 = 7.0e6
    pa = 101325.0
    gamma = 1.2
    expansion = 8.0
    pe = p0 * pressure_ratio(mach_from_area_ratio(expansion, gamma), gamma)
    cf_from_ratio = thrust_coefficient(p0, pa, gamma, area_ratio=expansion)
    cf_from_pressure = thrust_coefficient(p0, pa, gamma, exit_pressure=pe)
    assert cf_from_ratio == pytest.approx(cf_from_pressure, rel=1e-9)


def test_exit_area_is_throat_times_area_ratio():
    assert exit_area(0.01, 4.0) == 0.04
    assert exit_area(2.5e-3, 1.0) == 2.5e-3


def test_throat_area_from_thrust_and_back():
    p0 = 6.0e6
    pa = 101325.0
    gamma = 1.3
    expansion = 10.0
    thrust = 25_000.0
    at = throat_area(thrust, p0, pa, gamma, area_ratio=expansion)
    cf = thrust_coefficient(p0, pa, gamma, area_ratio=expansion)
    assert at == pytest.approx(thrust / (p0 * cf))
    assert p0 * cf * at == pytest.approx(thrust)


def test_mass_flow_is_pc_at_over_cstar():
    p0 = 4.0e6
    t0 = 3200.0
    at = 0.002
    gamma = 1.25
    molar_mass = 0.022
    cstar = characteristic_velocity(t0, gamma, molar_mass)
    mdot = mass_flow_rate(p0, t0, at, gamma, molar_mass)
    assert mdot == pytest.approx(p0 * at / cstar)


def test_thrust_equals_mass_flow_times_effective_exhaust_velocity():
    p0 = 7.0e6
    t0 = 3500.0
    gamma = 1.2
    molar_mass = 0.022
    pa = 101325.0
    expansion = 12.0
    thrust = 50_000.0

    at = throat_area(thrust, p0, pa, gamma, area_ratio=expansion)
    ae = exit_area(at, expansion)
    mdot = mass_flow_rate(p0, t0, at, gamma, molar_mass)
    cf = thrust_coefficient(p0, pa, gamma, area_ratio=expansion)
    cstar = characteristic_velocity(t0, gamma, molar_mass)
    isp = specific_impulse(
        p0, t0, gamma, molar_mass, pa, area_ratio=expansion
    )
    v_eq = cf * cstar

    assert ae / at == pytest.approx(expansion)
    assert thrust == pytest.approx(mdot * v_eq)
    assert thrust == pytest.approx(mdot * isp * G0)
    assert isp == pytest.approx(v_eq / G0)
    assert isp == pytest.approx(cf * cstar / G0)


def test_specific_impulse_matches_exit_pressure_input():
    p0 = 5.0e6
    t0 = 2800.0
    gamma = 1.4
    molar_mass = 0.02
    pa = 0.0
    expansion = area_ratio(3.0, gamma)
    pe = p0 * pressure_ratio(3.0, gamma)
    isp_ratio = specific_impulse(
        p0, t0, gamma, molar_mass, pa, area_ratio=expansion
    )
    isp_pe = specific_impulse(
        p0, t0, gamma, molar_mass, pa, exit_pressure=pe
    )
    assert isp_ratio == pytest.approx(isp_pe, rel=1e-9)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        characteristic_velocity(2000.0, gamma=1.0, molar_mass=0.02)
    with pytest.raises(ValueError):
        characteristic_velocity(2000.0, gamma=1.4, molar_mass=0.0)
    with pytest.raises(ValueError):
        thrust_coefficient(1e6, 0.0, 1.4)
    with pytest.raises(ValueError):
        thrust_coefficient(
            1e6, 0.0, 1.4, exit_pressure=1e5, area_ratio=4.0
        )
    with pytest.raises(ValueError):
        thrust_coefficient(1e6, -1.0, 1.4, area_ratio=4.0)
    with pytest.raises(ValueError):
        exit_area(0.01, 0.5)
    with pytest.raises(ValueError):
        throat_area(0.0, 1e6, 0.0, 1.4, area_ratio=2.0)
    with pytest.raises(ValueError):
        mass_flow_rate(1e6, 2000.0, -0.1, 1.4, 0.02)
