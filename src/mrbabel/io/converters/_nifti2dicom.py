"""NIfTI to DICOM Conversion Utilities."""

__all__ = ["nifti2dicom"]

import copy
import json
import warnings

import numpy as np

import mrd
import nii2dcm.nii
import nii2dcm.svr
import pydicom

from nibabel import Nifti1Image
from nii2dcm.dcm_writer import (
    transfer_nii_hdr_series_tags,
    transfer_nii_hdr_instance_tags,
)
from pydicom.datadict import keyword_for_tag

from ._mrd2dicom import IMTYPE_MAPS


def nifti2dicom(nii_paths: list[str], nii: list[Nifti1Image]) -> list[pydicom.Dataset]:
    """Convert NIfTI to Dicom dataset."""
    # Get json
    json_paths = [".".join([path.split(".")[0], "json"]) for path in nii_paths]
    json_list = []
    for json_path in json_paths:
        with open(json_path) as json_file:
            json_list.append(json.loads(json_file.read()))

    # Get contrast indexes and image type
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

    # Get images
    img = [vol.get_fdata().astype(np.float32) for vol in nii]

    # Fill dicom list
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
            vendor = dicom.ds.get("Manufacturer", "default")
            if "GE" in vendor.upper():
                dicom.ds.ImageType.insert(2, IMTYPE_MAPS[imtype[idx].name]["default"])

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

    return dsets
