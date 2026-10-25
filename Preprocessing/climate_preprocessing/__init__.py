"""Strict monthly CF reading and calendar-aware complete-season reduction."""
from .archive import Coordinates, FieldSpec, MonthlyArchive, MonthlyField
from .seasons import SeasonalField, iter_seasons, seasonal_mean, month_days, season_dates
from .units import convert_units
from .io import write_seasonal, write_seasonal_stream

__version__ = "0.1.0"
__all__ = ["Coordinates", "FieldSpec", "MonthlyArchive", "MonthlyField", "SeasonalField",
           "iter_seasons", "seasonal_mean", "month_days", "season_dates", "convert_units", "write_seasonal", "write_seasonal_stream"]
