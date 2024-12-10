"""NIfTI reading routines."""

__all__ = ["read_nifti"]

import copy
import json
import multiprocessing
import warnings

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

import mrd
import nibabel as nib

import pydicom
from pydicom.datadict import keyword_for_tag

from ..._file_search import get_paths

from ...data import sort_images

import nii2dcm.nii
import nii2dcm.svr
from nii2dcm.dcm_writer import (
    transfer_nii_hdr_series_tags,
    transfer_nii_hdr_instance_tags,
)

from ..dicom._dicom2mrd import read_dicom_header, read_dicom_images
from ..dicom._mrd2dicom import IMTYPE_MAPS


def read_nifti(
    paths: list[str], sort: bool = True
) -> tuple[mrd.ImageArray, mrd.Header]:
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
    image : mrd.Image Array | list[mrd.Image]
        MRD ImageArray  or list of MRD Images parsed from NIfTI files.
    head : mrd.Head
        MRD Header parsed from NIfTI files.

    """
    if isinstance(paths, str) and paths.endswith(".dcm"):
        nii_paths = [paths]
    else:
        nii_paths = get_paths("nii", paths, ext2="nii.gz")
    with ThreadPool(multiprocessing.cpu_count()) as pool:
        nii = pool.map(nib.load, nii_paths)

    # get json
    json_paths = [".".join([path.split(".")[0], "json"]) for path in nii_paths]
    json_list = []
    for json_path in json_paths:
        with open(json_path) as json_file:
            json_list.append(json.loads(json_file.read()))

    # get contrast indexes and image type
    name_length = np.asarray([len(path.split("_")) for path in nii_paths])
    is_magnitude = name_length == name_length.min()
    contrast_idx = []
    imtype = []
    for n in range(len(nii_paths)):
        if is_magnitude[n]:
            _contrast_idx = nii_paths[n].split("_")[-1]
            _contrast_idx = _contrast_idx.split(".")[0]
            _contrast_idx = int(_contrast_idx[1:]) - 1
            contrast_idx.append(_contrast_idx)
            imtype.append(mrd.ImageType.MAGNITUDE)
        else:
            _contrast_idx = nii_paths[n].split("_")[-2]
            _contrast_idx = int(_contrast_idx[1:]) - 1
            contrast_idx.append(_contrast_idx)
            _imtype = nii_paths[n].split("_")[-1]
            _imtype = _imtype.split(".")[0]
            if "real" in _imtype.lower():
                imtype.append(mrd.ImageType.REAL)
            if "imag" in _imtype.lower():
                imtype.append(mrd.ImageType.IMAG)
            if "ph" in _imtype.lower():
                imtype.append(mrd.ImageType.PHASE)

    # get images
    img = [vol.get_fdata().astype(np.float32) for vol in nii]

    # fill dicom list
    dsets = []
    instance_idx = 0
    for idx in range(len(nii)):
        nii2dcm_parameters = nii2dcm.nii.Nifti.get_nii2dcm_parameters(nii[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dicom = nii2dcm.dcm.DicomMRI("nii2dcm_dicom_mri.dcm")
            transfer_nii_hdr_series_tags(dicom, nii2dcm_parameters)
            dicom.ds.BitsAllocated = 32

        # update from json
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tag in dicom.ds.keys():
                keyword = keyword_for_tag(tag)
                if keyword in json_list[idx]:
                    setattr(dicom.ds, keyword, json_list[idx][keyword])
                    
            # update image type
            ImageType = dicom.ds.ImageType
            ImageType = ImageType[:2] + [IMTYPE_MAPS[imtype[idx].name]["default"]] + ImageType[-1]
            dicom.ds.ImageType = ImageType

            for instance_index in range(0, nii2dcm_parameters["NumberOfInstances"]):
                transfer_nii_hdr_instance_tags(
                    dicom, nii2dcm_parameters, instance_index
                )
                setattr(dicom.ds, "InstanceNumber", instance_idx)

                # Instance UID – unique to current slice
                dicom.ds.SOPInstanceUID = pydicom.uid.generate_uid(None)

                # Write pixel data
                dicom.ds.FloatPixelData = img[idx][:, :, instance_index].tobytes()

                # append
                dsets.append(copy.deepcopy(dicom.ds))
                instance_idx += 1

    # initialize header
    head = read_dicom_header(dsets[0])

    # convert dicoms to MRD
    images, head = read_dicom_images(dsets, head)
    if sort:
        return sort_images(images), head

    return images, head
