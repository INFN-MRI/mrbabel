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

from . import fidall  # noqa

from .fidall import *  # noqa

__all__.extend(fidall.__all__)

from . import gehc  # noqa

from .gehc import *  # noqa

__all__.extend(gehc.__all__)

from . import _image_builder  # noqa

from ._image_builder import *  # noqa

__all__.extend(_image_builder.__all__)
