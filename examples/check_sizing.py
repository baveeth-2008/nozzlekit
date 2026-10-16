"""Example: size a nozzle from chamber conditions, area ratio, and thrust."""

from nozzlekit.isentropic import mach_from_area_ratio, pressure_ratio
from nozzlekit.sizing import (
    characteristic_velocity,
    exit_area,
    mass_flow_rate,
    specific_impulse,
    throat_area,
    thrust_coefficient,
)

chamber_pressure = 2.0e6  # Pa (2 MPa)
chamber_temperature = 3500.0  # K
gamma = 1.2
molar_mass = 0.022  # kg/mol (22 kg/kmol)
expansion_ratio = 8.0
ambient_pressure = 101325.0  # Pa
thrust = 1000.0  # N

exit_mach = mach_from_area_ratio(expansion_ratio, gamma, supersonic=True)
exit_pressure = chamber_pressure * pressure_ratio(exit_mach, gamma)
cstar = characteristic_velocity(chamber_temperature, gamma, molar_mass)
cf = thrust_coefficient(
    chamber_pressure, ambient_pressure, gamma, area_ratio=expansion_ratio
)
at = throat_area(
    thrust, chamber_pressure, ambient_pressure, gamma, area_ratio=expansion_ratio
)
ae = exit_area(at, expansion_ratio)
mdot = mass_flow_rate(chamber_pressure, chamber_temperature, at, gamma, molar_mass)
isp = specific_impulse(
    chamber_pressure,
    chamber_temperature,
    gamma,
    molar_mass,
    ambient_pressure,
    area_ratio=expansion_ratio,
)

print(f"exit Mach number: {exit_mach} (dimensionless)")
print(f"exit pressure: {exit_pressure} Pa")
print(f"c*: {cstar} m/s")
print(f"thrust coefficient CF: {cf} (dimensionless)")
print(f"throat area: {at} m^2")
print(f"exit area: {ae} m^2")
print(f"mass flow rate: {mdot} kg/s")
print(f"Isp: {isp} s")
