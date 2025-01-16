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

from . import ismrmrd  # noqa

from .ismrmrd import *  # noqa

__all__.extend(ismrmrd.__all__)

from . import siemens  # noqa

from .siemens import *  # noqa

__all__.extend(siemens.__all__)
