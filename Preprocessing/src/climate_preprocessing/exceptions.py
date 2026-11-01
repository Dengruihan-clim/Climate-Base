"""Public validation errors, compatible with existing ValueError handlers.

Climate-Base improvement: distinguish input-contract failures without changing
the exception base class used by the existing monthly preprocessing API.
"""


class PreprocessingError(ValueError):
    """An input does not satisfy a preprocessing contract."""


class CoordinateError(PreprocessingError):
    """A coordinate is missing, ambiguous or physically inconsistent."""


class MetadataError(PreprocessingError):
    """CF metadata contains conflicting or dangling references."""


class TimeCoordinateError(CoordinateError):
    """A decoded time coordinate is invalid or unordered."""
