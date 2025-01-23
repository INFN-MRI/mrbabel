"""Main API."""

__all__ = []

try:
    from mrbabel_data import testdata
except Exception:

    def testdata(name):  # noqa
        raise ImportError(
            "Not implemented! Please install mrbabel-data (pip install mrbabel-data)"
        )


__all__.append("testdata")

from . import io  # noqa
from . import utils  # noqa
