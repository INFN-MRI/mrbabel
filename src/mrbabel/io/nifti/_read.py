"""NIfTI reading routines."""

__all__ = ["read_nifti"]


import multiprocessing

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

import mrd
import nibabel as nib

from ..._file_search import get_paths

from ..converters._dicom2mrd import read_dicom_header, read_dicom_images
from ..converters._nifti2dicom import nifti2dicom
from ..sorting import sort_images


def read_nifti(
    paths: list[str], sort: bool = True
) -> tuple[mrd.ImageArray | list[mrd.Image], mrd.Header]:
    """
    Read a list of NIfTI files using a thread pool.

    Parameters
    ----------
    paths : str | list of str
        Path or list of file paths to NIfTI files.
    sort : bool, optional
        If ``True``, sort list of MRD Images into a MRD ImageArray.
        The default is ``True``.

    Returns
    -------
    image : mrd.ImageArray | list[mrd.Image]
        MRD ImageArray  or list of MRD Images parsed from NIfTI files.
    head : mrd.Head
        MRD Header parsed from NIfTI files.

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
