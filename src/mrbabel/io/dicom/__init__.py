"""DICOM read and write subroutines."""

__all__ = []

from ._read import read_dicom  # noqa

__all__.append("read_dicom")

from ._write import write_dicom  # noqa

__all__.append("write_dicom")
