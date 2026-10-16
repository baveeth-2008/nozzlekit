"""Ideal rocket-nozzle sizing for a calorically perfect gas.

Chamber (stagnation) quantities are denoted with a subscript c.  All inputs
and outputs use SI units: Pa, K, kg/mol, m², N, kg/s, m/s, and seconds for
specific impulse.

The relations come from one-dimensional isentropic nozzle theory as given in
Sutton & Biblarz, *Rocket Propulsion Elements*, and Anderson, *Modern
Compressible Flow*.  Mach number, area ratio, and static-to-stagnation ratios
are delegated to :mod:`nozzlekit.isentropic`.

Either an exit pressure or an area ratio may be given; the other is recovered
from the isentropic relations.
"""

from __future__ import annotations

from . import isentropic

# CODATA 2018 molar gas constant, J/(mol·K)
R_UNIVERSAL = 8.314462618
# Standard gravitational acceleration used to report Isp in seconds, m/s²
G0 = 9.80665


def _check_gamma(gamma: float) -> None:
    if gamma <= 1.0:
        raise ValueError(f"gamma must be greater than 1, got {gamma}")


def _check_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _gas_constant(molar_mass: float) -> float:
    """Specific gas constant R = R_u / M, in J/(kg·K).

    R_u is the molar gas constant 8.314462618 J/(mol·K) (CODATA 2018).
    ``molar_mass`` is therefore in kg/mol (for example 0.028 kg/mol, not 28).
    """
    _check_positive("molar_mass", molar_mass)
    return R_UNIVERSAL / molar_mass


def _mach_from_pressure_ratio(pressure_ratio: float, gamma: float) -> float:
    """Invert p / p_c = (1 + (gamma - 1)/2 M²)^(-gamma/(gamma-1)).

    The closed form is

        M = sqrt( 2/(gamma - 1) * ((p_c / p)^((gamma - 1)/gamma) - 1) )

    which is the algebraic inverse of :func:`isentropic.pressure_ratio`.
    """
    if pressure_ratio <= 0.0 or pressure_ratio > 1.0:
        raise ValueError(
            f"exit/chamber pressure ratio must be in (0, 1], got {pressure_ratio}"
        )
    if pressure_ratio == 1.0:
        return 0.0
    return (
        2.0
        / (gamma - 1.0)
        * (pressure_ratio ** ((1.0 - gamma) / gamma) - 1.0)
    ) ** 0.5


def _exit_state(
    chamber_pressure: float,
    gamma: float,
    exit_pressure: float | None,
    area_ratio: float | None,
) -> tuple[float, float, float]:
    """Return (exit Mach, p_e / p_c, A_e / A*)."""
    _check_gamma(gamma)
    _check_positive("chamber_pressure", chamber_pressure)
    if (exit_pressure is None) == (area_ratio is None):
        raise ValueError("provide exactly one of exit_pressure or area_ratio")

    if area_ratio is not None:
        mach = isentropic.mach_from_area_ratio(area_ratio, gamma, supersonic=True)
        return mach, isentropic.pressure_ratio(mach, gamma), area_ratio

    _check_positive("exit_pressure", exit_pressure)
    p_ratio = exit_pressure / chamber_pressure
    mach = _mach_from_pressure_ratio(p_ratio, gamma)
    if mach == 0.0:
        raise ValueError("exit pressure equal to chamber pressure gives no nozzle flow")
    return mach, p_ratio, isentropic.area_ratio(mach, gamma)


def characteristic_velocity(
    chamber_temperature: float,
    gamma: float,
    molar_mass: float,
) -> float:
    """Characteristic velocity c* in m/s.

    c* = sqrt(gamma R T_c) / (gamma * (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1))))

    where R = R_u / M.  Equivalently c* = p_c A* / m_dot, so the sonic-throat
    mass flux is p_c / c*.

    Source: Sutton & Biblarz, *Rocket Propulsion Elements* (characteristic
    velocity / throat mass-flow relation); the same sonic mass flux appears in
    Anderson, *Modern Compressible Flow*.
    """
    _check_gamma(gamma)
    _check_positive("chamber_temperature", chamber_temperature)
    gas_constant = _gas_constant(molar_mass)
    sonic_factor = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    return (gamma * gas_constant * chamber_temperature) ** 0.5 / (gamma * sonic_factor)


def thrust_coefficient(
    chamber_pressure: float,
    ambient_pressure: float,
    gamma: float,
    *,
    exit_pressure: float | None = None,
    area_ratio: float | None = None,
) -> float:
    """Thrust coefficient C_F, including the (p_e - p_a) A_e pressure term.

    C_F = sqrt(
              (2 gamma² / (gamma - 1))
              * (2 / (gamma + 1))^((gamma + 1) / (gamma - 1))
              * (1 - (p_e / p_c)^((gamma - 1) / gamma))
          ) + ((p_e - p_a) / p_c) * (A_e / A*)

    The square-root term is the ideal momentum contribution at the given
    expansion; the second term is the pressure thrust.  It vanishes for
    optimal expansion (p_e = p_a) and is positive in vacuum.

    Thrust is then F = C_F p_c A*.

    Source: Sutton & Biblarz, *Rocket Propulsion Elements* (thrust coefficient
    with pressure term).  p_e / p_c and A_e / A* come from
    :func:`isentropic.pressure_ratio` and :func:`isentropic.area_ratio`
    (or the inverse of the pressure ratio if p_e is the given input).
    """
    if ambient_pressure < 0.0:
        raise ValueError(f"ambient_pressure cannot be negative, got {ambient_pressure}")
    _mach, p_ratio, expansion_ratio = _exit_state(
        chamber_pressure, gamma, exit_pressure, area_ratio
    )
    momentum = (
        (2.0 * gamma**2 / (gamma - 1.0))
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        * (1.0 - p_ratio ** ((gamma - 1.0) / gamma))
    ) ** 0.5
    return momentum + (p_ratio - ambient_pressure / chamber_pressure) * expansion_ratio


def throat_area(
    thrust: float,
    chamber_pressure: float,
    ambient_pressure: float,
    gamma: float,
    *,
    exit_pressure: float | None = None,
    area_ratio: float | None = None,
) -> float:
    """Throat area A* in m² from a desired thrust.

    A* = F / (p_c C_F)

    which is the definition F = C_F p_c A* rearranged (Sutton & Biblarz,
    *Rocket Propulsion Elements*).  C_F includes the ambient-pressure term,
    so sea-level and vacuum throats differ for the same vacuum-equivalent
    nozzle.
    """
    _check_positive("thrust", thrust)
    cf = thrust_coefficient(
        chamber_pressure,
        ambient_pressure,
        gamma,
        exit_pressure=exit_pressure,
        area_ratio=area_ratio,
    )
    return thrust / (chamber_pressure * cf)


def exit_area(throat: float, area_ratio: float) -> float:
    """Exit area A_e in m² from the throat area and the area ratio.

    A_e = (A_e / A*) A*

    Source: definition of the nozzle area ratio (Sutton & Biblarz, *Rocket
    Propulsion Elements*; Anderson, *Modern Compressible Flow*).
    """
    _check_positive("throat area", throat)
    if area_ratio < 1.0:
        raise ValueError(f"A/A* cannot be less than 1, got {area_ratio}")
    return throat * area_ratio


def mass_flow_rate(
    chamber_pressure: float,
    chamber_temperature: float,
    throat: float,
    gamma: float,
    molar_mass: float,
) -> float:
    """Choked mass flow rate m_dot in kg/s.

    m_dot = p_c A* / c*

    which is the definition of characteristic velocity solved for mass flow
    (Sutton & Biblarz, *Rocket Propulsion Elements*).  Equivalently, with
    R = R_u / M,

    m_dot = A* p_c gamma / sqrt(gamma R T_c)
            * (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1)))

    the sonic-throat mass flux of Anderson, *Modern Compressible Flow*.
    """
    _check_positive("chamber_pressure", chamber_pressure)
    _check_positive("throat area", throat)
    return (
        chamber_pressure
        * throat
        / characteristic_velocity(chamber_temperature, gamma, molar_mass)
    )


def specific_impulse(
    chamber_pressure: float,
    chamber_temperature: float,
    gamma: float,
    molar_mass: float,
    ambient_pressure: float,
    *,
    exit_pressure: float | None = None,
    area_ratio: float | None = None,
    g0: float = G0,
) -> float:
    """Specific impulse I_sp in seconds.

    I_sp = C_F c* / g_0 = F / (m_dot g_0)

    so the effective exhaust velocity is v_eq = C_F c* = I_sp g_0 and thrust
    satisfies F = m_dot v_eq.  g_0 defaults to 9.80665 m/s².

    Source: Sutton & Biblarz, *Rocket Propulsion Elements* (relation between
    specific impulse, thrust coefficient, and characteristic velocity).
    """
    _check_positive("g0", g0)
    cf = thrust_coefficient(
        chamber_pressure,
        ambient_pressure,
        gamma,
        exit_pressure=exit_pressure,
        area_ratio=area_ratio,
    )
    return cf * characteristic_velocity(chamber_temperature, gamma, molar_mass) / g0
