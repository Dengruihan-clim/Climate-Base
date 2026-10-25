"""Generate analytical fields and runnable configs for five diagnostics.

All fields are synthetic. Method references are available in
climate_dynamics.citations and REFERENCES.md.
Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def create_demo(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    radius, omega = 6_371_000., 7.292115e-5
    lat, lon = np.linspace(-90., 90., 25), np.arange(48) * 7.5
    phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    shape = (len(lat), len(lon))
    psi = 2e7 * np.cos(phi) * np.sin(lam)
    chi = 3e7 * np.cos(phi) * np.cos(lam)
    u_background = np.broadcast_to(20 * np.cos(phi), shape).copy()
    v_background = np.zeros(shape)
    perturbation = {
        "eta": -2 * psi / radius**2, "div": -2 * chi / radius**2,
        "udiv": np.broadcast_to(-3e7 / radius * np.sin(lam), shape).copy(),
        "vdiv": -3e7 / radius * np.sin(phi) * np.cos(lam),
        "eta_x": np.broadcast_to(-4e7 / radius**3 * np.cos(lam), shape).copy(),
        "eta_y": 4e7 / radius**3 * np.sin(phi) * np.sin(lam),
    }
    background = {
        "eta": np.broadcast_to(2 * (omega + 20/radius) * np.sin(phi), shape).copy(),
        "div": np.zeros(shape), "udiv": np.zeros(shape), "vdiv": np.zeros(shape),
        "eta_x": np.zeros(shape),
        "eta_y": np.broadcast_to(2 * (omega + 20/radius) / radius * np.cos(phi), shape).copy(),
    }
    np.savez(destination / "wave_inputs.npz", lat=lat, lon=lon, psi=psi,
             u_background=u_background, v_background=v_background)
    np.savez(destination / "source_inputs.npz", lat=lat, lon=lon,
             **{key: background[key] + perturbation[key] for key in background})
    np.savez(destination / "linear_inputs.npz", lat=lat, lon=lon,
             **{"background_" + key: value for key, value in background.items()},
             **{"perturbation_" + key: value for key, value in perturbation.items()})
    urot = 2e7 / radius * np.sin(phi) * np.sin(lam)
    vrot = np.broadcast_to(2e7 / radius * np.cos(lam), shape).copy()
    np.savez(destination / "wind_inputs.npz", lat=lat, lon=lon,
             u=u_background + perturbation["udiv"] + urot,
             v=v_background + perturbation["vdiv"] + vrot)
    mass_lat, mass_lon = np.arange(-60., 61., 30.), np.arange(0., 360., 30.)
    pressure = np.array([3000., 10000., 30000., 50000., 70000., 100000.])
    v = ((1 + np.arange(3)[:, None, None, None] / 10)
         * (1 + pressure[None, :, None, None]/1e5)
         * np.sin(2*np.deg2rad(mass_lat))[None, None, :, None])
    v = np.broadcast_to(v, (3, len(pressure), len(mass_lat), len(mass_lon))).copy()
    np.savez(destination / "mass_inputs.npz", lat=mass_lat, lon=mass_lon, pressure=pressure, v=v)
    rws_units = {"eta": "s-1", "div": "s-1", "udiv": "m s-1", "vdiv": "m s-1",
                 "eta_x": "s-1 m-1", "eta_y": "s-1 m-1"}
    configs = {
        "tn_waf": {"input": "wave_inputs.npz", "units": {"psi": "m2 s-1", "u_background": "m s-1", "v_background": "m s-1"},
                   "options": {"pressure_hpa": 200., "min_speed": 5., "min_abs_lat": 10.}},
        "rossby_wave_source": {"input": "source_inputs.npz", "units": rws_units},
        "linear_rws": {"input": "linear_inputs.npz", "units": rws_units},
        "wind": {"input": "wind_inputs.npz", "units": {"u": "m s-1", "v": "m s-1"},
                 "options": {"truncation": 8, "gridtype": "regular"}},
        "mass_streamfunction": {"input": "mass_inputs.npz", "units": {"v": "m s-1", "pressure": "Pa"},
                                "options": {"cyclic": True, "min_fraction": 1., "reference_pressure_pa": 3000.}},
    }
    for method, config in configs.items():
        config.update(method=method, output=method + "_result.npz")
        (destination / (method + ".json")).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    create_demo(args.output_dir)
    print("Created synthetic arrays and five method configurations.")


if __name__ == "__main__":
    main()
