"""Small explicit scalar checks shared by calendar and archive APIs."""
import numpy as np


def checked_integer(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, excluding booleans")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return result


def checked_bool(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)
