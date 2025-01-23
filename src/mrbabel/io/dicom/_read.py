"""DICOM reading routines."""

__all__ = ["read_dicom"]

import pydicom
import multiprocessing

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import mrd

from ..._file_search import get_paths


from ..converters._dicom2mrd import read_dicom_header, read_dicom_images
from ..sorting import sort_images


def read_dicom(
    paths: list[str], sort: bool = True
) -> tuple[mrd.ImageArray, mrd.Header]:
    """
    Read a list of DICOM files using a thread pool.

    Parameters
    ----------
    paths : str | list of str
        Path to DICOM folder or list of file paths to DICOM files.
        Supports wildcard.
    sort : bool, optional
        If ``True``, sort list of MRD Images into a MRD ImageArray.
        The default is ``True``.

    Returns
    -------
    image : mrd.ImageArray | list[mrd.Image]
        MRD ImageArray  or list of MRD Images parsed from DICOM files.
    head : mrd.Head
        MRD Header parsed from DICOM files.

    """
    if isinstance(paths, str) and paths.endswith(".dcm") or paths.endswith(".IMA"):
        paths = [paths]
    else:
        paths = get_paths("dcm", paths, ext2="IMA")
    if len(paths) == 0:
        raise ValueError("DICOM files not found in target directory.")

    def dcmread(filename):
        return pydicom.dcmread(
            filename,
            force=False,
        )

    with ThreadPool(multiprocessing.cpu_count()) as pool:
        dsets = pool.map(dcmread, paths)

    # Initialize header
    head = read_dicom_header(dsets[0])

    # Read images
    images, head = read_dicom_images(dsets, head)

    if sort:
        image = sort_images(images, head)
        image.data = np.flip(image.data, -3)
        return image, head

    return images, head
