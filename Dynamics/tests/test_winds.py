"""Wind input/adapter contracts plus separately identified real-backend checks.

Mocked tests below verify validation, shapes, order and argument forwarding;
they do NOT validate spherical-harmonic mathematics. The real-backend test
calls the standalone analytic runner and is explicitly skipped only when the
optional backend cannot be imported. The runner itself fails on missing
dependencies, allowing an independent real-backend verification report.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from climate_dynamics import winds


class ContractWind:
    """Manufactured outputs for adapter contracts, not spherical harmonics."""
    instances = []
    bad_output = None

    def __init__(self, u, v, **options):
        self.u, self.v, self.options = u, v, options
        self.calls = []
        self.__class__.instances.append(self)

    def sfvp(self, **kwargs):
        self.calls.append(("sfvp", kwargs))
        psi = self.u + 10.
        if self.bad_output == "shape":
            psi = psi[:-1]
        elif self.bad_output == "nan":
            psi = psi.copy()
            psi.flat[0] = np.nan
        elif self.bad_output == "masked":
            psi = np.ma.array(psi, mask=False)
            psi.mask.flat[0] = True
        elif self.bad_output == "complex":
            psi = psi.astype(complex) + 1j
        return psi, self.v + 20.

    def vrtdiv(self, **kwargs):
        self.calls.append(("vrtdiv", kwargs))
        return self.u + 30., self.v + 40.

    def helmholtz(self, **kwargs):
        self.calls.append(("helmholtz", kwargs))
        return self.u * .1, self.v * .2, self.u * .9, self.v * .8

    def absolutevorticity(self, **kwargs):
        self.calls.append(("absolutevorticity", kwargs))
        self.absolute = self.u + kwargs["omega"]
        return self.absolute

    def gradient(self, scalar, **kwargs):
        assert scalar is self.absolute
        self.calls.append(("gradient", kwargs))
        return scalar + 100., scalar + 200.


@pytest.fixture
def contract_backend(monkeypatch):
    ContractWind.instances = []
    ContractWind.bad_output = None
    monkeypatch.setattr(winds, "_vector_wind_class", lambda: ContractWind)
    return ContractWind


def grid():
    lat = np.linspace(-90., 90., 7)
    lon = np.arange(12) * 30.
    u = np.arange(84., dtype=float).reshape(7, 12)
    return lat, lon, u, -u


def test_contract_calls_public_operations_once_with_common_settings(contract_backend):
    lat, lon, u, v = grid()
    result = winds.diagnose(u, v, lat, lon, truncation=3, radius=7e6, omega=0., legfunc="computed")
    backend = contract_backend.instances[0]
    np.testing.assert_array_equal(backend.u, u[::-1])
    np.testing.assert_array_equal(backend.v, v[::-1])
    assert backend.u.flags.c_contiguous and backend.u.dtype == np.float64
    assert backend.options == {"gridtype": "regular", "rsphere": 7e6, "legfunc": "computed"}
    assert [name for name, _ in backend.calls] == ["sfvp", "vrtdiv", "helmholtz", "absolutevorticity", "gradient"]
    assert all(kwargs["truncation"] == 3 for _, kwargs in backend.calls)
    assert backend.calls[3][1]["omega"] == 0.
    np.testing.assert_array_equal(result["psi"], u + 10.)
    np.testing.assert_array_equal(result["chi"], v + 20.)
    np.testing.assert_array_equal(result["eta_x"], u + 100.)
    np.testing.assert_array_equal(result["eta_y"], u + 200.)
    np.testing.assert_allclose(result["u_total"], u)
    np.testing.assert_allclose(result["v_total"], v)
    assert result["uchi"] is result["udiv"] and result["vchi"] is result["vdiv"]
    assert result["units"]["eta_y"] == "s-1 m-1"
    assert result["radius"] == 7e6 and result["omega"] == 0.
    assert result["latitude_order"] == "ascending"
    np.testing.assert_array_equal(result["lat"], lat)
    np.testing.assert_array_equal(result["lon"], lon)


def test_contract_latitude_reversal_preserves_northward_sign(contract_backend):
    lat, lon, u, v = grid()
    a = winds.diagnose(u, v, lat, lon, truncation=3)
    b = winds.diagnose(u[::-1], v[::-1], lat[::-1], lon, truncation=3)
    for name in winds.FIELD_UNITS:
        np.testing.assert_array_equal(a[name], b[name])
    np.testing.assert_array_equal(b["lat"], lat)


def test_contract_gaussian_grid_and_float32_coordinate_rounding(contract_backend):
    lat = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(8)[0])).astype(np.float32)
    lon = np.arange(16) * 22.5
    u = np.arange(128.).reshape(8, 16)
    result = winds.diagnose(u, -u, lat, lon, truncation=3, gridtype="gaussian")
    assert contract_backend.instances[0].options["gridtype"] == "gaussian"
    np.testing.assert_array_equal(result["lat"], lat)
    np.testing.assert_array_equal(result["vdiv"], -.2*u)


@pytest.mark.parametrize("leading", [(2,), (2, 3), (1, 1)])
def test_contract_leading_dimensions_are_independent_and_restored(contract_backend, leading):
    lat, lon, u, v = grid()
    count = int(np.prod(leading))
    data = (np.arange(count)[:, None, None]*1000 + u).reshape((*leading, *u.shape))
    result = winds.diagnose(data, -data, lat, lon, truncation=2)
    backend = contract_backend.instances[0]
    assert backend.u.shape == (len(lat), len(lon), count)
    np.testing.assert_array_equal(backend.u, data[..., ::-1, :].reshape(count, 7, 12).transpose(1, 2, 0))
    np.testing.assert_array_equal(result["psi"], data + 10.)
    np.testing.assert_array_equal(result["chi"], -data + 20.)
    assert result["eta_y"].shape == data.shape


def test_contract_none_truncation_retains_backend_native_choice(contract_backend):
    lat, lon, u, v = grid()
    result = winds.diagnose(u, v, lat, lon, truncation=None)
    assert result["truncation"] is None
    assert all(kwargs["truncation"] is None for _, kwargs in contract_backend.instances[0].calls)


@pytest.mark.parametrize("kind", ["shape", "nan", "masked", "complex"])
def test_contract_invalid_backend_products_fail_explicitly(contract_backend, kind):
    contract_backend.bad_output = kind
    lat, lon, u, v = grid()
    with pytest.raises(RuntimeError, match="windspharm returned"):
        winds.diagnose(u, v, lat, lon, truncation=2)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_missing_winds_fail_before_backend_import(monkeypatch, bad):
    lat, lon, u, v = grid()
    u[2, 3] = bad
    monkeypatch.setattr(winds, "_vector_wind_class", lambda: pytest.fail("backend must not be imported"))
    with pytest.raises(ValueError, match="entirely finite"):
        winds.diagnose(u, v, lat, lon, truncation=2)


def test_masked_and_complex_winds_are_not_filled_or_silently_cast(contract_backend):
    lat, lon, u, v = grid()
    masked = np.ma.array(u, mask=False)
    masked.mask[0, 0] = True
    with pytest.raises(ValueError, match="entirely finite"):
        winds.diagnose(masked, v, lat, lon, truncation=2)
    with pytest.raises(ValueError, match="real numerical"):
        winds.diagnose(u.astype(complex) + 1j, v, lat, lon, truncation=2)
    assert not contract_backend.instances


def test_regional_repeated_and_out_of_bounds_coordinates_are_rejected(contract_backend):
    lat, lon, u, v = grid()
    with pytest.raises(ValueError, match="include both poles"):
        winds.diagnose(u[1:-1], v[1:-1], lat[1:-1], lon, truncation=2)
    with pytest.raises(ValueError, match="complete equally spaced"):
        winds.diagnose(u[:, :-1], v[:, :-1], lat, lon[:-1], truncation=2)
    with pytest.raises(ValueError, match="duplicated endpoint"):
        winds.diagnose(np.pad(u, ((0, 0), (0, 1))), np.pad(v, ((0, 0), (0, 1))), lat, np.r_[lon, 360.], truncation=2)
    bad_lat = np.array([-90.000005, -45., 0., 45., 90.000005])
    with pytest.raises(ValueError, match="Latitude"):
        winds.diagnose(np.zeros((5, 12)), np.zeros((5, 12)), bad_lat, lon, truncation=2)
    near_poles = np.array([-89.999995, -45., 0., 45., 89.999995])
    with pytest.raises(ValueError, match="both poles exactly"):
        winds.diagnose(np.zeros((5, 12)), np.zeros((5, 12)), near_poles, lon, truncation=2)
    assert not contract_backend.instances


@pytest.mark.parametrize("kwargs", [{"truncation": True}, {"truncation": 0}, {"truncation": 6},
                                    {"radius": 0}, {"radius": np.nan}, {"omega": np.inf},
                                    {"omega": True}, {"legfunc": "other"}, {"gridtype": "other"}])
def test_bad_parameters_are_rejected_before_backend(contract_backend, kwargs):
    lat, lon, u, v = grid()
    with pytest.raises(ValueError):
        winds.diagnose(u, v, lat, lon, **({"truncation": 2} | kwargs))
    assert not contract_backend.instances


def test_shape_mismatch_or_empty_leading_axis_is_rejected(contract_backend):
    lat, lon, u, v = grid()
    for bad_u, bad_v in ((u[None], v), (u[:, :-1], v[:, :-1]), (np.empty((0, 7, 12)), np.empty((0, 7, 12)))):
        with pytest.raises(ValueError, match="identical shape"):
            winds.diagnose(bad_u, bad_v, lat, lon, truncation=2)
    assert not contract_backend.instances


def test_optional_backend_error_is_explicit_without_numerical_fallback(monkeypatch):
    lat, lon, u, v = grid()
    def unavailable():
        raise ImportError("optional windspharm backend unavailable")
    monkeypatch.setattr(winds, "_vector_wind_class", unavailable)
    with pytest.raises(ImportError, match="windspharm"):
        winds.diagnose(u, v, lat, lon, truncation=2)


def test_real_backend_synthetic_analytic_runner_when_installed():
    pytest.importorskip("windspharm.standard", reason="REAL windspharm backend unavailable; contract tests are not physical validation")
    path = Path(__file__).resolve().parents[1] / "tools" / "validate_winds_backend.py"
    spec = importlib.util.spec_from_file_location("real_wind_backend_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.validate_backend()
    assert result["status"] == "passed"
    assert result["analytic_cases_passed"] == 14
    assert all(case["passed"] for case in result["cases"])
