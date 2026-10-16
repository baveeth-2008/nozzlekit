# nozzlekit

Open-source Python toolkit for rocket nozzle design: isentropic sizing, bell contour generation, and CAD export.

> **Status: early development (v0.0.1).** Only the isentropic flow module exists so far. The rest is planned, see the roadmap below.

## What works today

The `nozzlekit.isentropic` module implements the standard one-dimensional isentropic flow relations for a calorically perfect gas:

- `area_ratio(mach, gamma)`: A / A*
- `mach_from_area_ratio(ratio, gamma, supersonic=True)`: inverse of the above, both branches
- `pressure_ratio`, `temperature_ratio`, `density_ratio`: static to stagnation ratios

```python
from nozzlekit import area_ratio, mach_from_area_ratio, pressure_ratio

# Exit Mach number of a nozzle with an area ratio of 8, gamma = 1.2
exit_mach = mach_from_area_ratio(8.0, gamma=1.2)
exit_pressure_ratio = pressure_ratio(exit_mach, gamma=1.2)  # p_exit / p_chamber
```

## Install and test

```bash
git clone https://github.com/baveeth-2008/nozzlekit.git
cd nozzlekit
pip install -e ".[dev]"
pytest
```

## Validation

The isentropic relations are tested against the standard gamma = 1.4 isentropic flow table (NACA Report 1135, Anderson's *Modern Compressible Flow*), plus consistency checks (ideal gas law, round trips between Mach number and area ratio). A fuller validation page will be added as more modules are built.

## Roadmap

- [x] Isentropic flow relations
- [ ] Nozzle sizing: throat area, exit area, thrust, mass flow, Isp, c*
- [ ] Bell nozzle contour (Rao parabola approximation)
- [ ] Contour export (CSV and DXF) for CAD software such as Fusion 360
- [ ] Command-line interface
- [ ] Validation against worked examples in *Rocket Propulsion Elements* (Sutton and Biblarz)
- [ ] Optional OpenFOAM case generator for CFD comparison

## Scope and sources

This project uses only textbook-level, publicly available methods and data.

## License

MIT, see [LICENSE](LICENSE).
