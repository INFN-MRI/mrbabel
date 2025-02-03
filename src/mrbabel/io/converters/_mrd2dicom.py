"""MRD to DICOM Conversion Utilities."""

__all__ = [
    "DEFAULTS",
    "IMTYPE_MAPS",
    "dump_dicom_images",
]

import base64
import re
import warnings

import mrd
import numpy as np
import pydicom

from ...utils import get_user_param

# Defaults for input arguments
DEFAULTS = {
    "out_group": "dataset",
}

# Lookup table for image types, configurable by vendor
IMTYPE_MAPS = {
    "MAGNITUDE": {"default": "M", "GE": 0},
    "PHASE": {"default": "P", "GE": 1},
    "REAL": {"default": "R", "GE": 2},
    "IMAG": {"default": "I", "GE": 3},
}


def _convert_patient_position(PatientPosition):
    return "".join(PatientPosition.name.split("_"))


def dump_dicom_images(
    images: list[mrd.Image],
    head: mrd.Header,
) -> list[mrd.Acquisition]:
    """Create list of DICOM files from MRD Header and a list of MRD images."""
    return [_dump_dicom_image(image, head) for image in images]


def _dump_dicom_image(image, head):
    image_data = image.data
    image_head = image.head
    image_meta = image.meta

    # Use previously JSON serialized header as a starting point, if available
    if image_meta.get("DicomJson") is not None:
        dset = pydicom.dataset.Dataset.from_json(
            base64.b64decode(image_meta["DicomJson"])
        )
    else:
        dset = pydicom.dataset.Dataset()

    # Enforce explicit little endian for written DICOM files
    dset.file_meta = pydicom.dataset.FileMetaDataset()
    dset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dset.file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    dset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    pydicom.dataset.validate_file_meta(dset.file_meta)

    # ----- Update DICOM header from MRD header -----
    # fill patient information
    try:
        if head.subject_information is None:
            pass
        else:
            if head.subject_information.patient_name is not None:
                dset.PatientName = head.subject_information.patient_name
            if head.subject_information.patient_weight_kg is not None:
                dset.PatientWeight = head.subject_information.patient_weight_kg
            if head.subject_information.patient_height_m is not None:
                dset.PatientHeight = head.subject_information.patient_height_m
            if head.subject_information.patient_id is not None:
                dset.PatientID = head.subject_information.patient_id
            if head.subject_information.patient_birthdate is not None:
                dset.PatientBirthDate = head.subject_information.patient_birthdate
            if head.subject_information.patient_gender is not None:
                dset.PatientSex = head.subject_information.patient_gender.name
    except Exception:
        warnings.warn(
            "Unable to set header information from MRD header's patient_information section",
            UserWarning,
        )

    # fill study information
    try:
        if head.study_information is None:
            pass
        else:
            if head.study_information.study_date is not None:
                dset.StudyDate = head.study_information.study_date
            if head.study_information.study_time is not None:
                dset.StudyTime = head.study_information.study_time
            if head.study_information.study_id is not None:
                dset.StudyID = head.study_information.study_id
            if head.study_information.study_description is not None:
                dset.StudyDescription = head.study_information.study_description
            if head.study_information.study_instance_uid is not None:
                dset.StudyInstanceUID = head.study_information.study_instance_uid
    except Exception:
        warnings.warn(
            "Unable to set header information from MRD header's study_information section",
            UserWarning,
        )

    # fill measurement information
    try:
        if head.measurement_information is None:
            pass
        else:
            if head.measurement_information.measurement_id is not None:
                dset.SeriesInstanceUID = head.measurement_information.measurement_id
            if head.measurement_information.patient_position is not None:
                dset.PatientPosition = _convert_patient_position(
                    head.measurement_information.patient_position
                )
            if head.measurement_information.protocol_name is not None:
                dset.SeriesDescription = head.measurement_information.protocol_name
            if head.measurement_information.frame_of_reference_uid is not None:
                dset.FrameOfReferenceUID = (
                    head.measurement_information.frame_of_reference_uid
                )
    except Exception:
        warnings.warn(
            "Unable to set header information from MRD header's measurement_information section",
            UserWarning,
        )

    # fill acquisition system information
    try:
        if head.acquisition_system_information is None:
            pass
        else:
            if head.acquisition_system_information.system_vendor is not None:
                dset.Manufacturer = head.acquisition_system_information.system_vendor
            if head.acquisition_system_information.system_model is not None:
                dset.ManufacturerModelName = (
                    head.acquisition_system_information.system_model
                )
            if head.acquisition_system_information.system_field_strength_t is not None:
                dset.MagneticFieldStrength = (
                    head.acquisition_system_information.system_field_strength_t
                )
            if head.acquisition_system_information.institution_name is not None:
                dset.InstitutionName = (
                    head.acquisition_system_information.institution_name
                )
            if head.acquisition_system_information.station_name is not None:
                dset.StationName = head.acquisition_system_information.station_name
    except Exception:
        warnings.warn(
            "Unable to set information from MRD header's acquisition_system_information section",
            UserWarning,
        )

    # fill experimental condition information
    dset.MagneticFieldStrength = (
        head.experimental_conditions.h1resonance_frequency_hz / 4258e4
    )

    # Set dset pixel image_data from MRD Image image_data
    dset.Rows = image_data.shape[-2]
    dset.Columns = image_data.shape[-1]

    if (image_data.dtype == "uint16") or (image_data.dtype == "int16"):
        dset.PixelData = np.squeeze(
            image_data
        ).tobytes()  # image_data is [cha z y x] -- squeeze to [y x] for [row col]
        dset.BitsAllocated = 16
        dset.BitsStored = 16
        dset.HighBit = 15
    elif (
        (image_data.dtype == "uint32")
        or (image_data.dtype == "int32")
        or (image_data.dtype == "int")
    ):
        dset.PixelData = np.squeeze(
            image_data
        ).tobytes()  # image_data is [cha z y x] -- squeeze to [y x] for [row col]
        dset.BitsAllocated = 32
        dset.BitsStored = 32
        dset.HighBit = 31
    elif image_data.dtype == "float32":
        dset.FloatPixelData = np.squeeze(
            image_data
        ).tobytes()  # image_data is [cha z y x] -- squeeze to [y x] for [row col]
        dset.BitsAllocated = 32
        dset.BitsStored = 32
        dset.HighBit = 31
    elif image_data.dtype == "float64":
        dset.DoubleFloatPixelData = np.squeeze(
            image_data
        ).tobytes()  # image_data is [cha z y x] -- squeeze to [y x] for [row col]
        dset.BitsAllocated = 64
        dset.BitsStored = 64
        dset.HighBit = 63
    else:
        print("Unsupported image_data type: ", image_data.dtype)

    dset.SeriesNumber = image_head.image_series_index
    dset.InstanceNumber = image_head.image_index

    # ----- Set some mandatory default values -----
    if not "SamplesPerPixel" in dset:
        dset.SamplesPerPixel = 1

    if not "PhotometricInterpretation" in dset:
        dset.PhotometricInterpretation = "MONOCHROME2"

    if not "PixelRepresentation" in dset:
        dset.PixelRepresentation = 0  # Unsigned integer

    if not "ImageType" in dset:
        dset.ImageType = ["ORIGINAL", "PRIMARY", "M"]

    if not "SeriesNumber" in dset:
        dset.SeriesNumber = 1

    if not "SeriesDescription" in dset:
        dset.SeriesDescription = ""

    if not "InstanceNumber" in dset:
        dset.InstanceNumber = 1

    # ----- Update DICOM header from MRD ImageHeader -----
    try:
        if "GE" in dset.Manufacturer:
            vendor = "GE"
        else:
            vendor = "default"
    except Exception:
        vendor = "default"
    dset.ImageType[2] = IMTYPE_MAPS[image_head.image_type.name][vendor]
    dset.PixelSpacing = [
        float(image_head.field_of_view[0]) / image_data.shape[-2],
        float(image_head.field_of_view[1]) / image_data.shape[-1],
    ]
    dset.SliceThickness = image_head.field_of_view[2]
    dset.ImagePositionPatient = [
        image_head.position[0],
        image_head.position[1],
        image_head.position[2],
    ]
    dset.ImageOrientationPatient = [
        image_head.line_dir[0],
        image_head.line_dir[1],
        image_head.line_dir[2],
        image_head.col_dir[0],
        image_head.col_dir[1],
        image_head.col_dir[2],
    ]

    time_sec = image_head.acquisition_time_stamp / 1000 / 2.5
    hour = int(np.floor(time_sec / 3600))
    min = int(np.floor((time_sec - hour * 3600) / 60))
    sec = time_sec - hour * 3600 - min * 60
    dset.AcquisitionTime = "%02.0f%02.0f%09.6f" % (hour, min, sec)
    # dset.TriggerTime = image_head.physiology_time_stamp[0] / 2.5

    # ----- Update DICOM header from MRD Image MetaAttributes -----
    if image_meta.get("SeriesDescription") is not None:
        dset.SeriesDescription = image_meta["SeriesDescription"]

    if image_meta.get("SeriesDescriptionAdditional") is not None:
        dset.SeriesDescription = (
            dset.SeriesDescription + image_meta["SeriesDescriptionAdditional"]
        )

    if image_meta.get("ImageComment") is not None:
        dset.ImageComment = "_".join(image_meta["ImageComment"])

    if image_meta.get("ImageType") is not None:
        dset.ImageType = image_meta["ImageType"]

    if (image_meta.get("ImageRowDir") is not None) and (
        image_meta.get("ImageColumnDir") is not None
    ):
        dset.ImageOrientationPatient = [
            float(image_meta["ImageRowDir"][0]),
            float(image_meta["ImageRowDir"][1]),
            float(image_meta["ImageRowDir"][2]),
            float(image_meta["ImageColumnDir"][0]),
            float(image_meta["ImageColumnDir"][1]),
            float(image_meta["ImageColumnDir"][2]),
        ]

    if image_meta.get("RescaleIntercept") is not None:
        dset.RescaleIntercept = image_meta["RescaleIntercept"]
    else:
        dset.RescaleIntercept = 0.0

    if image_meta.get("RescaleSlope") is not None:
        dset.RescaleSlope = image_meta["RescaleSlope"]
    else:
        dset.RescaleSlope = 1.0

    if image_meta.get("WindowCenter") is not None:
        dset.WindowCenter = image_meta["WindowCenter"]

    if image_meta.get("WindowWidth") is not None:
        dset.WindowWidth = image_meta["WindowWidth"]

    # setting sequence parameters
    if head.sequence_parameters is not None:
        idx = image_head.contrast
        if any(head.sequence_parameters.flip_angle_deg):
            dset.FlipAngle = head.sequence_parameters.flip_angle_deg[idx]
        if any(head.sequence_parameters.t_r):
            dset.RepetitionTime = head.sequence_parameters.t_r[idx]
        if any(head.sequence_parameters.t_e):
            dset.EchoTime = head.sequence_parameters.t_e[idx]
        if any(head.sequence_parameters.t_i):
            dset.InversionTime = head.sequence_parameters.t_i[idx]

    # setting spacing
    fov_z = head.encoding[-1].encoded_space.field_of_view_mm.z
    nz = head.encoding[-1].encoded_space.matrix_size.z
    dset.SpacingBetweenSlices = fov_z / nz

    # setting imaging mode
    if get_user_param(head, "ImagingMode"):
        imode = get_user_param(head, "ImagingMode")
        if "3" in imode:
            imode = "3D"
        else:
            imode = "2D"
        dset.MRAcquisitionType = imode

    return dset


def _parse_custom_dictionary(file_path):
    custom_tags = {}
    if file_path is not None:
        with open(file_path, "r") as f:
            for line in f:
                match = re.match(
                    r"\(([\dA-F]{4}),([\dA-F]{4})\)\s+(\w+)\s+(.+)", line.strip()
                )
                if match:
                    group, element, vr, name = match.groups()
                    tag = (int(group, 16), int(element, 16))
                    custom_tags[tag] = (vr, name)
    return custom_tags
