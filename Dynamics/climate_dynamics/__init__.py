"""Atmospheric dynamical diagnostics with original-method references.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
See REFERENCES.md and each kernel's docstring for scientific attribution.
"""
from .waves import linear_rws, rossby_wave_source, tn_waf
from .winds import diagnose as diagnose_wind
from .circulation import fixed_longitude_mean, longitude_cell_weights, mass_streamfunction
from .citations import CITATIONS, METHOD_REFERENCES

__version__ = "0.1.0"
__all__ = ["tn_waf", "rossby_wave_source", "linear_rws", "diagnose_wind",
           "mass_streamfunction", "longitude_cell_weights", "fixed_longitude_mean",
           "CITATIONS", "METHOD_REFERENCES"]
