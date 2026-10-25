"""Synthetic analytic checks using the REAL optional windspharm backend.

No climate data, model business entry points, filesystem products, or mocked
transforms are used. Missing dependencies return a failing status, not a set
of skipped tests. Backend/package versions and analytic case counts are
reported without local environment paths.
"""
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

if "__file__" in globals():
    package_root = Path(__file__).resolve().parents[1]
    if (package_root / "climate_dynamics").is_dir():
        sys.path.insert(0, str(package_root))

from climate_dynamics.winds import EARTH_OMEGA_S, EARTH_RADIUS_M, diagnose


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "metadata unavailable"


def validate_backend():
    # Explicitly require the actual backend; this runner never substitutes it.
    from windspharm.standard import VectorWind  # noqa: F401
    import spharm

    cases = []

    def close(actual, expected, *, rtol=3e-5, atol=0):
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=rtol, atol=atol)

    for gridtype in ("regular", "gaussian"):
        lat = (np.linspace(-90., 90., 25) if gridtype == "regular"
               else np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(24)[0])))
        lon = np.arange(48) * 7.5
        phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
        shape = (len(lat), len(lon))
        zero = np.zeros(shape)
        settings = {"truncation": 8, "gridtype": gridtype}
        f = 2 * EARTH_OMEGA_S * np.sin(phi)
        beta = 2 * EARTH_OMEGA_S / EARTH_RADIUS_M * np.cos(phi)

        result = diagnose(zero, zero, lat, lon, **settings)
        for name in ("psi", "chi", "vort", "div", "urot", "vrot", "udiv", "vdiv"):
            close(result[name], 0., rtol=0, atol=1e-20)
        close(result["eta"], f, atol=3e-9)
        close(result["eta_x"], 0., rtol=0, atol=3e-14)
        close(result["eta_y"], beta, rtol=4e-4, atol=3e-14)
        cases.append({"grid": gridtype, "case": "zero wind retains planetary vorticity and physical beta", "passed": True})

        speed = 10.
        rotation_u = np.broadcast_to(speed * np.cos(phi), shape).copy()
        rotating = diagnose(rotation_u, zero, lat, lon, **settings)
        close(rotating["psi"], -EARTH_RADIUS_M * speed * np.sin(phi), atol=1500.)
        close(rotating["chi"], 0., rtol=0, atol=800.)
        close(rotating["vort"], 2 * speed / EARTH_RADIUS_M * np.sin(phi), atol=3e-10)
        close(rotating["div"], 0., rtol=0, atol=3e-10)
        close(rotating["urot"], rotation_u, atol=1e-4)
        for name in ("udiv", "vdiv", "vrot"):
            close(rotating[name], 0., rtol=0, atol=1e-4)
        effective = EARTH_OMEGA_S + speed / EARTH_RADIUS_M
        close(rotating["eta"], 2 * effective * np.sin(phi), atol=3e-9)
        close(rotating["eta_y"], 2 * effective / EARTH_RADIUS_M * np.cos(phi), rtol=4e-4, atol=3e-14)
        cases.append({"grid": gridtype, "case": "solid-body rotation signs and Helmholtz components", "passed": True})

        amplitude = 3e7
        potential = amplitude * np.cos(phi) * np.cos(lam)
        u = np.broadcast_to(-amplitude / EARTH_RADIUS_M * np.sin(lam), shape).copy()
        v = -amplitude / EARTH_RADIUS_M * np.sin(phi) * np.cos(lam)
        divergent = diagnose(u, v, lat, lon, **settings)
        close(divergent["chi"], potential, atol=800.)
        close(divergent["psi"], 0., rtol=0, atol=800.)
        close(divergent["udiv"], u, atol=1e-4)
        close(divergent["vdiv"], v, atol=1e-4)
        close(divergent["div"], -2 * potential / EARTH_RADIUS_M**2, atol=3e-10)
        for name in ("urot", "vrot"):
            close(divergent[name], 0., rtol=0, atol=1e-4)
        close(divergent["eta_y"], beta, rtol=4e-4, atol=3e-14)
        cases.append({"grid": gridtype, "case": "degree-one velocity potential and physical divergence", "passed": True})

        perturbation = diagnose(u, v, lat, lon, omega=0., **settings)
        for name in ("eta", "vort"):
            close(perturbation[name], 0., rtol=0, atol=5e-10)
        for name in ("eta_x", "eta_y"):
            close(perturbation[name], 0., rtol=0, atol=3e-14)
        close(divergent["eta"] - perturbation["eta"], f, atol=3e-9)
        cases.append({"grid": gridtype, "case": "omega zero excludes planetary terms from perturbation diagnostics", "passed": True})

        reversed_result = diagnose(u[::-1], v[::-1], lat[::-1], lon, **settings)
        np.testing.assert_array_equal(reversed_result["lat"], lat)
        np.testing.assert_array_equal(reversed_result["lon"], lon)
        for name in ("psi", "chi", "vort", "div", "eta", "eta_x", "eta_y", "udiv", "vdiv", "urot", "vrot"):
            np.testing.assert_array_equal(reversed_result[name], divergent[name])
        cases.append({"grid": gridtype, "case": "latitude reversal preserves northward signs and ascending outputs", "passed": True})

        stacked = diagnose(np.stack((u, 2*u))[:, None], np.stack((v, 2*v))[:, None],
                           lat, lon, omega=0., **settings)
        assert stacked["chi"].shape == (2, 1, *shape)
        for index, scale in enumerate((1., 2.)):
            close(stacked["chi"][index, 0], scale * potential, atol=1600.)
            close(stacked["udiv"][index, 0], scale * u, atol=2e-4)
            close(stacked["vdiv"][index, 0], scale * v, atol=2e-4)
            close(stacked["div"][index, 0], -2 * scale * potential / EARTH_RADIUS_M**2, atol=6e-10)
        cases.append({"grid": gridtype, "case": "multiple leading dimensions preserve independent fields", "passed": True})

        custom_radius = 7e6
        custom = diagnose(rotation_u, zero, lat, lon, radius=custom_radius, omega=0., **settings)
        close(custom["psi"], -custom_radius * speed * np.sin(phi), atol=1500.)
        close(custom["eta"], 2 * speed / custom_radius * np.sin(phi), atol=3e-10)
        close(custom["eta_y"], 2 * speed / custom_radius**2 * np.cos(phi), rtol=4e-4, atol=3e-14)
        cases.append({"grid": gridtype, "case": "configured sphere radius controls physical inverse and gradient units", "passed": True})

    return {"status": "passed", "verification": "REAL windspharm/pyspharm synthetic analytic checks; no mocked transforms",
            "analytic_cases_passed": len(cases), "cases": cases,
            "versions": {"numpy": np.__version__, "windspharm": _version("windspharm"),
                         "spharm": str(getattr(spharm, "__version__", _version("pyspharm")))},
            "limitations": ["Analytic spherical fields only; no real reanalysis data were processed",
                            "Tolerances account for common single-precision SPHEREPACK builds",
                            "Contract tests with a mocked backend are separate from this report"]}


def main():
    try:
        result = validate_backend()
    except ImportError as error:
        print(json.dumps({"status": "dependency_unavailable", "backend_validated": False,
                          "reason": str(error), "analytic_cases_passed": 0}))
        return 2
    except Exception as error:
        print(json.dumps({"status": "failed", "backend_validated": False,
                          "error_type": type(error).__name__, "reason": str(error)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
