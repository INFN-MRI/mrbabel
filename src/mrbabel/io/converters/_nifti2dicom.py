"""NIfTI to DICOM Conversion Utilities."""

__all__ = ["nifti2dicom"]

import copy
import json as jsonlib
import warnings

import numpy as np

import mrd
import nii2dcm.dcm
import nii2dcm.nii
import nii2dcm.svr
import pydicom

from nibabel import Nifti1Image
from nii2dcm.dcm_writer import (
    transfer_nii_hdr_series_tags,
    transfer_nii_hdr_instance_tags,
)
from pydicom.datadict import keyword_for_tag

from ...utils._geometry import detect_scan_orientation
from ...utils._geometry import reorient_nifti

from ._mrd2dicom import IMTYPE_MAPS


def nifti2dicom(nii_paths: list[str], nii: list[Nifti1Image]) -> list[pydicom.Dataset]:
    """Convert NIfTI to Dicom dataset."""
    # Get json
    json_paths = [".".join([*path.split(".")[:-1], "json"]) for path in nii_paths]
    json = []
    for path in json_paths:
        with open(path) as file:
            json.append(jsonlib.loads(file.read()))

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

    # Get parameters from first volume
    idx = 0
    try:
        nii[idx].affine[:2] *= -1
        ndims = int(json[idx]["MRAcquisitionType"][0])
        iop = np.asarray(json[idx]["ImageOrientationPatientDICOM"]).reshape(2, 3)
        scan_orient = detect_scan_orientation(iop)
        if ndims == 2:
            if scan_orient == "ax":
                nii[idx] = reorient_nifti(nii[idx], "RAS")
            if scan_orient == "cor":  # not sure here
                nii[idx] = reorient_nifti(nii[idx], "RIA")
            if scan_orient == "sag":
                nii[idx] = reorient_nifti(nii[idx], "AIL")
        if ndims == 3:
            if scan_orient == "ax":
                nii[idx] = reorient_nifti(nii[idx], "RAI")
            if scan_orient == "cor":
                nii[idx] = reorient_nifti(nii[idx], "RIP")
            if scan_orient == "sag":
                nii[idx] = reorient_nifti(nii[idx], "AIR")

    except Exception:
        pass

    nii2dcm_parameters = nii2dcm.nii.Nifti.get_nii2dcm_parameters(nii[idx])

    # Fix wrong nii2dcm iop
    iop = -1 * np.asarray(nii2dcm_parameters["ImageOrientationPatient"]).reshape(2, 3)
    iop = np.flip(iop, axis=0).ravel()
    nii2dcm_parameters["ImageOrientationPatient"] = iop.tolist()

    # Fix wrong nii2dcm ipp (TODO: open PR and fix)
    nii2dcm_parameters = calc_ipp(nii[idx], nii2dcm_parameters)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dicom = nii2dcm.dcm.DicomMRI("nii2dcm_dicom_mri.dcm")
        transfer_nii_hdr_series_tags(dicom, nii2dcm_parameters)
        dicom.ds.BitsAllocated = 32

    # Update from json
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tag in dicom.ds.keys():
            keyword = keyword_for_tag(tag)
            if keyword in json[idx]:
                setattr(dicom.ds, keyword, json[idx][keyword])

    # Update with first instance params
    instance_index = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transfer_nii_hdr_instance_tags(dicom, nii2dcm_parameters, instance_index)
        setattr(dicom.ds, "InstanceNumber", instance_index)

    # Instance UID – unique to current slice
    dicom.ds.SOPInstanceUID = pydicom.uid.generate_uid(None)

    # Get nifti data and header
    nii_data = np.stack(img, axis=-1).T  # (others, (phases), nz, ny, nx)
    nii_head = {
        "json": json,
        "dset": dicom.ds,
        "ImagePositionPatient": nii2dcm_parameters["ImagePositionPatient"],
    }

    return nii_data, nii_head

    # Fill dicom list
    # dsets = []
    # instance_idx = 0
    # for idx in range(len(nii)):
    #     try:
    #         nii[idx].affine[:2] *= -1
    #         ndims = int(json[idx]["MRAcquisitionType"][0])
    #         iop = np.asarray(json[idx]["ImageOrientationPatientDICOM"]).reshape(
    #             2, 3
    #         )
    #         scan_orient = detect_scan_orientation(iop)
    #         if ndims == 2:
    #             if scan_orient == "ax":
    #                 nii[idx] = reorient_nifti(nii[idx], "RAS")
    #             if scan_orient == "cor":  # not sure here
    #                 nii[idx] = reorient_nifti(nii[idx], "RIA")
    #             if scan_orient == "sag":
    #                 nii[idx] = reorient_nifti(nii[idx], "AIL")
    #         if ndims == 3:
    #             if scan_orient == "ax":
    #                 nii[idx] = reorient_nifti(nii[idx], "RAI")
    #             if scan_orient == "cor":
    #                 nii[idx] = reorient_nifti(nii[idx], "RIP")
    #             if scan_orient == "sag":
    #                 nii[idx] = reorient_nifti(nii[idx], "AIR")

    #     except Exception:
    #         pass

    #     nii2dcm_parameters = nii2dcm.nii.Nifti.get_nii2dcm_parameters(nii[idx])

    #     # fix wrong nii2dcm iop
    #     iop = -1 * np.asarray(nii2dcm_parameters["ImageOrientationPatient"]).reshape(
    #         2, 3
    #     )
    #     iop = np.flip(iop, axis=0).ravel()
    #     nii2dcm_parameters["ImageOrientationPatient"] = iop.tolist()

    #     # fix wrong nii2dcm ipp (TODO: open PR and fix)
    #     nii2dcm_parameters = calc_ipp(nii[idx], nii2dcm_parameters)

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     dicom = nii2dcm.dcm.DicomMRI("nii2dcm_dicom_mri.dcm")
    #     transfer_nii_hdr_series_tags(dicom, nii2dcm_parameters)
    #     dicom.ds.BitsAllocated = 32

    # # update from json
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     for tag in dicom.ds.keys():
    #         keyword = keyword_for_tag(tag)
    #         if keyword in json[idx]:
    #             setattr(dicom.ds, keyword, json[idx][keyword])

    #     # update image type
    #     vendor = dicom.ds.get("Manufacturer", "default")
    #     if "GE" in vendor.upper():
    #         dicom.ds.ImageType.insert(2, IMTYPE_MAPS[imtype[idx].name]["default"])

    #     for instance_index in range(0, nii2dcm_parameters["NumberOfInstances"]):
    #         transfer_nii_hdr_instance_tags(
    #             dicom, nii2dcm_parameters, instance_index
    #         )
    #         setattr(dicom.ds, "InstanceNumber", instance_idx)

    #         # Instance UID – unique to current slice
    #         dicom.ds.SOPInstanceUID = pydicom.uid.generate_uid(None)

    #         # Write pixel data
    #         dicom.ds.FloatPixelData = img[idx][:, :, instance_index].tobytes()

    #         # append
    #         dsets.append(copy.deepcopy(dicom.ds))
    #         instance_idx += 1


def fnT1N(A, N):
    # Subfn: calculate T1N vector
    # A = affine matrix [4x4]
    # N = slice number (counting from 0)
    T1N = A.dot([0, 0, N, 1])
    return T1N


def calc_ipp(nib_nii, nii2dcm_parameters):
    A = nib_nii.affine
    image_pos_patient_array = nii2dcm_parameters["ImagePositionPatient"]
    nInstances = len(image_pos_patient_array)

    for iInstance in range(0, nInstances):
        T1N = fnT1N(A, iInstance)
        image_pos_patient_array[iInstance] = [T1N[0], T1N[1], T1N[2]]

    nii2dcm_parameters["ImagePositionPatient"] = image_pos_patient_array
    return nii2dcm_parameters
