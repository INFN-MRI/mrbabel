"""NIfTI reading routines."""

__all__ = ["read_nifti"]

import glob
import multiprocessing

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

import mrd
import nibabel as nib

from ..._file_search import get_paths

from ..converters._nifti2mrd import read_nifti_header, read_nifti_image
from ..converters._nifti2dicom import nifti2dicom
from ..sorting import unsort_images


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
        nii_paths = glob.glob(paths)
    else:
        nii_paths = get_paths("nii", paths, ext2="nii.gz")
    if len(nii_paths) == 0:
        raise ValueError("NIfTI files not found in target directory.")
    with ThreadPool(multiprocessing.cpu_count()) as pool:
        nii = pool.map(nib.load, nii_paths)

    # Convert NIfTI to Dicom
    nii_data, nii_head = nifti2dicom(nii_paths, nii)

    # Initialize header from Dicom
    nii_data, nii_head, head = read_nifti_header(nii_data, nii_head)

    # Convert dicoms to MRD
    image, head = read_nifti_image(nii_data, nii_head, head)

    if sort:
        return image, head
    else:
        image.data = np.flip(image.data, (-2, -1))
        images = unsort_images(image, head)

    return images, head