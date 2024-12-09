"""DICOM reading routines."""

__all__ = ["read_dicom"]

import pydicom
import multiprocessing

from multiprocessing.dummy import Pool as ThreadPool

import mrd

from ..._file_search import get_paths

from ...data import sort_images

from ._dicom2mrd import read_dicom_header, read_dicom_images


def read_dicom(paths: list[str]) -> tuple[mrd.ImageArray, mrd.Header]:
    """
    Read a list of DICOM files using a thread pool.

    Parameters
    ----------
    paths : list of str
        List of file paths to DICOM files.

    Returns
    -------
    image : list[mrd.Image]
        MRD ImageArray parsed from DICOM files.
    head : mrd.Head
        MRD Header parsed from DICOM files.

    """
    paths = get_paths("dcm", paths)
    with ThreadPool(multiprocessing.cpu_count()) as pool:
        dsets = pool.map(pydicom.dcmread, paths)

    # initialize header
    head = read_dicom_header(dsets[0])

    # read images
    images, head = read_dicom_images(dsets, head)

    # sort
    image = sort_images(images)

    return image, head
