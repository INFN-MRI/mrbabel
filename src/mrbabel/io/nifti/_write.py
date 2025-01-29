"""NIfTI writing routines."""

__all__ = ["write_nifti"]


import multiprocessing

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

import mrd
import nibabel as nib

from ..._file_search import get_paths

from ..converters._dicom2mrd import read_dicom_header, read_dicom_images
from ..converters._nifti2dicom import nifti2dicom
from ..sorting import sort_images


def write_nifti(
    path: str,
    image: mrd.ImageArray,
    head: mrd.Header,
):
    """
    path : str
        Path to output DICOM folder.
    image : mrd.ImageArray
        Input MRD ImageArray to be written.
    head : mrd.Header
        Input MRD ImageHeader to be written.

    """
    if isinstance(paths, str) and paths.endswith(".nii") or paths.endswith(".nii.gz"):
        nii_paths = [paths]
    else:
        nii_paths = get_paths("nii", paths, ext2="nii.gz")
    if len(nii_paths) == 0:
        raise ValueError("NIfTI files not found in target directory.")
    with ThreadPool(multiprocessing.cpu_count()) as pool:
        nii = pool.map(nib.load, nii_paths)

    # Convert NIfTI to Dicom
    dsets = nifti2dicom(nii_paths, nii)

    # Initialize header
    head = read_dicom_header(dsets[0])

    # Convert dicoms to MRD
    images, head = read_dicom_images(dsets, head)
    head = _convert_sequence_parameters_units(head)

    if sort:
        image, head = sort_images(images, head)
        image.data = np.flip(image.data.swapaxes(-1, -2), (-2, -1))
        return image, head

    return images, head


def _convert_sequence_parameters_units(head: mrd.Header) -> mrd.Header:
    if head.sequence_parameters:
        if head.sequence_parameters.t_r.any():
            head.sequence_parameters.t_r *= 1000.0
        if head.sequence_parameters.t_e.any():
            head.sequence_parameters.t_e *= 1000.0
        if head.sequence_parameters.t_i.any():
            if np.isclose(head.sequence_parameters.t_i, -1).any():
                head.sequence_parameters.t_i = []
            else:
                head.sequence_parameters.t_i *= 1000.0
    return head
