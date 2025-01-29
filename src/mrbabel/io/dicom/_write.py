"""DICOM writing routines."""

__all__ = ["write_dicom"]

import copy
import os

import pydicom
import multiprocessing

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import mrd

from ..converters._mrd2dicom import dump_dicom_images
from ..sorting import unsort_images


def write_dicom(
    path: str,
    image: mrd.ImageArray,
    head: mrd.Header,
):
    """
    Write MRD ImageArray to DICOM.

    Parameters
    ----------
    path : str
        Path to output DICOM folder.
    image : mrd.ImageArray
        Input MRD ImageArray to be written.
    head : mrd.Header
        Input MRD ImageHeader to be written.

    """
    images = unsort_images(image, head)

    # Broadcast sequence parameters to ncontrasts
    head = copy.deepcopy(head)
    head = _broadcast_sequence_parameters(head)

    # Create DICOM
    dsets = dump_dicom_images(images, head)

    # Create destination folder
    path = os.path.realpath(path)
    if os.path.exists(path) is False:
        os.makedirs(path)
    paths = [
        os.path.join(path, f"image-{str(n).zfill(4)}.dcm") for n in range(len(dsets))
    ]

    def dcmwrite(filename, dataset):
        pydicom.dcmwrite(
            filename,
            dataset,
            enforce_file_format=True,
            little_endian=True,
            implicit_vr=False,
        )

    # Writing
    paths_dsets = [(paths[n], dsets[n]) for n in range(len(dsets))]
    with ThreadPool(multiprocessing.cpu_count()) as pool:
        pool.starmap(dcmwrite, paths_dsets)


def _broadcast_sequence_parameters(head: mrd.Header) -> mrd.Header:
    params = {
        k: v
        for k, v in vars(head.sequence_parameters).items()
        if v is not None and len(v) > 0
    }
    params = dict(zip(params.keys(), np.broadcast_arrays(*params.values())))
    for k, v in params.items():
        setattr(head.sequence_parameters, k, v)

    return head
