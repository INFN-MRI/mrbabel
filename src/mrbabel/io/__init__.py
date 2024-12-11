"""I/O subroutines."""

__all__ = []

from . import dicom  # noqa

from .dicom import *  # noqa

__all__.extend(dicom.__all__)

from . import nifti  # noqa

from .nifti import *  # noqa

__all__.extend(nifti.__all__)

from . import mrd  # noqa

from .mrd import *  # noqa

__all__.extend(mrd.__all__)
