"""MRD to NIfTI Conversion Utilities."""

import copy

import numpy as np
import mrd

import nibabel as nib

from nibabel.orientations import axcodes2ornt, ornt_transform

from ..dicom._dicom2mrd import get_plane_normal
from ..dicom._mrd2dicom import dump_dicom_images
from ...data.operations import unsort_images


def dump_nifti_images(image: mrd.ImageArray, mrdhead: mrd.Header):
    iscomplex = np.iscomplexobj(image.data)

    # unsort and get dicom
    images = unsort_images(image)
    dsets = dump_dicom_images(images, mrdhead)
    json_base = _initialize_json_dict(dsets[0])

    # unpack header
    # affine = head.affine

    # resolution
    fov = mrdheader.encoding[-1].encoded_space
    dz = float(head.ref_dicom.SpacingBetweenSlices)
    dx, dy = head.ref_dicom.PixelSpacing
    resolution = np.asarray((dz, dy, dx))

    if head.TR is not None:
        TR = float(head.TR.min())
    else:
        TR = 1000.0

    if iscomplex:
        ...
    else:
        n_contrasts = image.data.shape[0]
        for n in range(n_contrasts):
            json_dict = copy.deepcopy(json_base)
            if mrdhead.sequence_parameters:
                json_dict["EchoNumber"] = n
                if mrdhead.sequence_parameters.flip_angle_deg:
                    json_dict["FlipAngle"] = mrdhead.sequence_parameters.flip_angle_deg
                if mrdhead.sequence_parameters.flip_angle_deg:
                    json_dict["EchoTime"] = mrdhead.sequence_parameters.t_e
                if mrdhead.sequence_parameters.flip_angle_deg:
                    json_dict["RepetitionTime"] = mrdhead.sequence_parameters.t_r
                if mrdhead.sequence_parameters.flip_angle_deg:
                    json_dict["InversionTime"] = mrdhead.sequence_parameters.t_i


def _initialize_json_dict(dicomDset):
    """Initialize Json dictionary."""
    json = {}

    if "SliceThickness" in dicomDset:
        json["SliceThickness"] = str(dicomDset.SliceThickness)
    if "SpacingBetweenSlices" in dicomDset:
        json["SpacingBetweenSlices"] = str(dicomDset.SpacingBetweenSlices)
    if "PatientName" in dicomDset:
        json["PatientName"] = str(dicomDset.PatientName)
    if "PatientWeight" in dicomDset:
        json["PatientWeight"] = str(dicomDset.PatientWeight)
    if "PatientID" in dicomDset:
        json["PatientID"] = str(dicomDset.PatientID)
    if "PatientBirthDate" in dicomDset:
        json["PatientBirthDate"] = str(dicomDset.PatientBirthDate)
    if "PatientAge" in dicomDset:
        json["PatientAge"] = str(dicomDset.PatientAge)
    if "PatientSex" in dicomDset:
        json["PatientSex"] = str(dicomDset.PatientSex)
    if "StudyDate" in dicomDset:
        json["StudyDate"] = str(dicomDset.StudyDate)
    if "StudyTime" in dicomDset:
        json["StudyTime"] = str(dicomDset.StudyTime)
    if "AccessionNumber" in dicomDset:
        json["AccessionNumber"] = str(dicomDset.AccessionNumber)
    if "ReferringPhysicianName" in dicomDset:
        json["ReferringPhysicianName"] = str(dicomDset.ReferringPhysicianName)
    if "StudyDescription" in dicomDset:
        json["StudyDescription"] = str(dicomDset.StudyDescription)
    if "StudyInstanceUID" in dicomDset:
        json["StudyInstanceUID"] = str(dicomDset.StudyInstanceUID)
    if "SeriesDate" in dicomDset:
        json["SeriesDate"] = str(dicomDset.SeriesDate)
    if "SeriesTime" in dicomDset:
        json["SeriesTime"] = str(dicomDset.SeriesTime)
    if "PatientPosition" in dicomDset:
        json["PatientPosition"] = str(dicomDset.PatientPosition)
    if "SequenceName" in dicomDset:
        json["SequenceName"] = str(dicomDset.SequenceName)
    if "FrameOfReferenceUID" in dicomDset:
        json["FrameOfReferenceUID"] = str(dicomDset.FrameOfReferenceUID)
    if "Manufacturer" in dicomDset:
        json["Manufacturer"] = str(dicomDset.Manufacturer)
    if "ManufacturerModelName" in dicomDset:
        json["ManufacturerModelName"] = str(dicomDset.ManufacturerModelName)
    if "MagneticFieldStrength" in dicomDset:
        json["MagneticFieldStrength"] = str(dicomDset.MagneticFieldStrength)
    if "InstitutionName" in dicomDset:
        json["InstitutionName"] = str(dicomDset.InstitutionName)
    if "StationName" in dicomDset:
        json["StationName"] = str(dicomDset.StationName)

    return json


def _make_nifti_affine(shape, position, orientation, resolution):
    """
    Return affine transform between voxel coordinates and mm coordinates.

    Parameters
    ----------
    shape : list
        volume shape (nz, ny, nx).
    resolution : list
        image resolution in mm (dz, dy, dz).
    position : list
        position of each slice (3, nz).
    orientation : list
        image orientation.

    Returns
    -------
    np.ndarray
        Affine matrix describing image position and orientation.

    Ref: https://nipy.org/nibabel/dicom/spm_dicom.html#spm-volume-sorting

    """
    # get image size
    nz, ny, nx = shape

    # get resoluzion
    dz, dy, dx = resolution

    # common parameters
    T = position
    T1 = T[:, 0].round(4)

    F = orientation
    dr, dc = np.asarray([dy, dx]).round(4)

    if nz == 1:  # single slice case
        n = get_plane_normal(orientation)
        ds = float(dz)

        A0 = np.stack(
            (
                np.append(F[0] * dc, 0),
                np.append(F[1] * dr, 0),
                np.append(-ds * n, 0),
                np.append(T1, 1),
            ),
            axis=1,
        )

    else:  # multi slice case
        N = nz
        TN = T[:, -1].round(4)
        A0 = np.stack(
            (
                np.append(F[0] * dc, 0),
                np.append(F[1] * dr, 0),
                np.append((TN - T1) / (N - 1), 0),
                np.append(T1, 1),
            ),
            axis=1,
        )

    # sign of affine matrix
    A0[:2, :] *= -1

    # reorient
    A = _reorient(shape, A0, "LAS")

    return A.astype(np.float32)


def _reorient(shape, affine, orientation):
    """Reorient input image to desired orientation."""
    orig_ornt = nib.io_orientation(affine)

    # get target orientation
    targ_ornt = axcodes2ornt(orientation)

    # estimate transform
    transform = ornt_transform(orig_ornt, targ_ornt)

    # reorient
    tmp = np.ones(shape[-3:], dtype=np.float32)
    tmp = nib.Nifti1Image(tmp, affine)
    tmp = tmp.as_reoriented(transform)

    return tmp.affine
