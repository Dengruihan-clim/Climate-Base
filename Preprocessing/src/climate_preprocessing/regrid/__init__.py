"""Optional xESMF/ESMF horizontal remapping; heavy imports occur only on use."""

from .api import regrid_dataset
from .grids import rectilinear_grid

__all__ = ["regrid_dataset", "rectilinear_grid"]
