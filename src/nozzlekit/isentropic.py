"""Isentropic flow relations for a calorically perfect gas.

These are the standard one-dimensional isentropic relations found in any
compressible-flow textbook (for example Anderson, *Modern Compressible Flow*,
or Sutton & Biblarz, *Rocket Propulsion Elements*).

Notation
--------
M      Mach number
gamma  ratio of specific heats (cp / cv), must be greater than 1
p0, T0, rho0   stagnation (chamber) pressure, temperature, density
A*     throat area, where M = 1
"""

from __future__ import annotations


def _check_gamma(gamma: float) -> None:
    if gamma <= 1.0:
        raise ValueError(f"gamma must be greater than 1, got {gamma}")


def _check_mach(mach: float) -> None:
    if mach < 0.0:
        raise ValueError(f"Mach number cannot be negative, got {mach}")


def temperature_ratio(mach: float, gamma: float = 1.4) -> float:
    """Static-to-stagnation temperature ratio, T / T0.

    T / T0 = 1 / (1 + (gamma - 1) / 2 * M^2)
    """
    _check_gamma(gamma)
    _check_mach(mach)
    return 1.0 / (1.0 + 0.5 * (gamma - 1.0) * mach**2)


def pressure_ratio(mach: float, gamma: float = 1.4) -> float:
    """Static-to-stagnation pressure ratio, p / p0.

    p / p0 = (1 + (gamma - 1) / 2 * M^2) ^ (-gamma / (gamma - 1))
    """
    _check_gamma(gamma)
    _check_mach(mach)
    return (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** (-gamma / (gamma - 1.0))


def density_ratio(mach: float, gamma: float = 1.4) -> float:
    """Static-to-stagnation density ratio, rho / rho0.

    rho / rho0 = (1 + (gamma - 1) / 2 * M^2) ^ (-1 / (gamma - 1))
    """
    _check_gamma(gamma)
    _check_mach(mach)
    return (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** (-1.0 / (gamma - 1.0))


def area_ratio(mach: float, gamma: float = 1.4) -> float:
    """Local area divided by throat area, A / A*.

    A / A* = (1 / M) * [ 2 / (gamma + 1) * (1 + (gamma - 1) / 2 * M^2) ]
                       ^ ((gamma + 1) / (2 * (gamma - 1)))

    The ratio is 1 at M = 1 and grows on both sides of it, so the same area
    ratio corresponds to two Mach numbers (one subsonic, one supersonic).
    """
    _check_gamma(gamma)
    if mach <= 0.0:
        raise ValueError(f"Mach number must be positive for an area ratio, got {mach}")
    base = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach**2)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return base**exponent / mach


def mach_from_area_ratio(
    ratio: float,
    gamma: float = 1.4,
    supersonic: bool = True,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Mach number for a given A / A*, found by bisection.

    Parameters
    ----------
    ratio:
        A / A*, must be at least 1.
    gamma:
        Ratio of specific heats.
    supersonic:
        True returns the supersonic solution (M >= 1, the nozzle exit),
        False returns the subsonic solution (M <= 1, the converging section).
    """
    _check_gamma(gamma)
    if ratio < 1.0:
        raise ValueError(f"A/A* cannot be less than 1, got {ratio}")
    if ratio == 1.0:
        return 1.0

    if supersonic:
        low, high = 1.0, 2.0
        while area_ratio(high, gamma) < ratio:  # grow the bracket until it contains the answer
            high *= 2.0
            if high > 1e6:
                raise ValueError("Area ratio too large to bracket a Mach number")
    else:
        low, high = 1e-9, 1.0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        value = area_ratio(mid, gamma)
        if abs(value - ratio) <= tol * ratio:
            return mid
        # Supersonic branch: A/A* increases with M. Subsonic branch: it decreases.
        if (value < ratio) == supersonic:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
