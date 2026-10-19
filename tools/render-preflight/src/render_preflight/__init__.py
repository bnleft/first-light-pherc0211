"""Storage-aware descriptive QA for surface-conditioned CT renders."""

from .scan import scan
from .storage import LocalSlab

__all__ = ["LocalSlab", "scan"]
__version__ = "0.1.0"
