"""MRD to DICOM Conversion Utilities."""

__all__ = [
    "DEFAULTS",
    "IMTYPE_MAPS",
    "dump_dicom_images",
]

import warnings
import base64

import mrd
import numpy as np
import pydicom

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


def dump_dicom_images(
    images: list[mrd.Image], mrdhead: mrd.Header
) -> list[mrd.Acquisition]:
    """Create list of DICOM files from MRD Header and a list of MRD images."""
    return [_dump_dicom_image(image, mrdhead) for image in images]


def _dump_dicom_image(image, mrdhead):

    data = image.data
    head = image.head
    meta = image.meta

    # Use previously JSON serialized header as a starting point, if available
    if meta.get("dicom_json") is not None:
        dset = pydicom.dataset.Dataset.from_json(base64.b64decode(meta["DicomJson"]))
    else:
        dset = pydicom.dataset.Dataset()

    # Enforce explicit little endian for written DICOM files
    dset.file_meta = pydicom.dataset.FileMetaDataset()
    dset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dset.file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    dset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    pydicom.dataset.validate_file_meta(dset.file_meta)

    # FileMetaInformationGroupLength is still missing?
    dset.is_little_endian = True
    dset.is_implicit_VR = False

    # ----- Update DICOM header from MRD header -----
    # fill patient information
    try:
        if mrdhead.patient_information is None:
            pass
        else:
            if mrdhead.patient_information.patient_name is not None:
                dset.PatientName = mrdhead.patient_information.patient_name
            if mrdhead.patient_information.weight_kg is not None:
                dset.PatientWeight = mrdhead.patient_information.weight_kg
            if mrdhead.patient_information.height_m is not None:
                dset.PatientHeight = mrdhead.patient_information.height_m
            if mrdhead.patient_information.patient_id is not None:
                dset.PatientID = mrdhead.patient_information.patient_id
            if mrdhead.patient_information.patient_birthdate is not None:
                dset.PatientBirthDate = mrdhead.patient_information.patient_birthdate
            if mrdhead.patient_information.patient_gender is not None:
                dset.PatientSex = mrdhead.patient_information.patient_gender
    except Exception:
        warnings.warn(
            "Unable to set header information from MRD header's patient_information section",
            UserWarning,
        )

    # fill study information
    try:
        if mrdhead.study_information is None:
            pass
        else:
            if mrdhead.study_information.study_date is not None:
                dset.StudyDate = mrdhead.study_information.study_date
            if mrdhead.study_information.study_time is not None:
                dset.StudyTime = mrdhead.study_information.study_time
            if mrdhead.study_information.study_id is not None:
                dset.StudyID = mrdhead.study_information.study_id
            if mrdhead.study_information.study_description is not None:
                dset.StudyDescription = mrdhead.study_information.study_description
            if mrdhead.study_information.study_instance_uid is not None:
                dset.StudyInstanceUID = mrdhead.study_information.study_instance_uid
    except Exception:
        pass
        # raise ValueError(
        #     "Error setting header information from MRD header's study_information section"
        # )

    # fill measurement information
    try:
        if mrdhead.measurement_information is None:
            pass
        else:
            if mrdhead.measurement_information.measurement_id is not None:
                dset.SeriesInstanceUID = mrdhead.measurement_information.measurement_id
            if mrdhead.measurement_information.patient_position is not None:
                dset.PatientPosition = (
                    mrdhead.measurement_information.patient_position.name
                )
            if mrdhead.measurement_information.protocolName is not None:
                dset.SeriesDescription = mrdhead.measurement_information.protocol_name
            if mrdhead.measurement_information.frame_of_reference_uid is not None:
                dset.FrameOfReferenceUID = (
                    mrdhead.measurement_information.frame_of_reference_uid
                )
    except Exception:
        warnings.warn(
            "Unable to set header information from MRD header's measurement_information section",
            UserWarning,
        )

    # fill acquisition system information
    try:
        if mrdhead.acquisition_system_information is None:
            pass
        else:
            if mrdhead.acquisition_system_information.system_vendor is not None:
                dset.Manufacturer = mrdhead.acquisition_system_information.system_vendor
            if mrdhead.acquisition_system_information.system_model is not None:
                dset.ManufacturerModelName = (
                    mrdhead.acquisition_system_information.system_model
                )
            if (
                mrdhead.acquisition_system_information.system_field_strength_t
                is not None
            ):
                dset.MagneticFieldStrength = (
                    mrdhead.acquisition_system_information.system_field_strength_t
                )
            if mrdhead.acquisition_system_information.institution_name is not None:
                dset.InstitutionName = (
                    mrdhead.acquisition_system_information.institution_name
                )
            if mrdhead.acquisition_system_information.station_name is not None:
                dset.StationName = mrdhead.acquisition_system_information.station_name
    except Exception:
        warnings.warn(
            "Unable to set information from MRD header's acquisition_system_information section",
            UserWarning,
        )

    # fill experimental condition information
    dset.MagneticFieldStrength = (
        mrdhead.experimental_conditions.h1resonance_frequency_hz / 4258e4
    )

    # Set dset pixel data from MRD Image data
    dset.PixelData = np.squeeze(
        data
    ).tobytes()  # data is [cha z y x] -- squeeze to [y x] for [row col]
    dset.Rows = data.shape[2]
    dset.Columns = data.shape[3]

    if (data.dtype == "uint16") or (data.dtype == "int16"):
        dset.BitsAllocated = 16
        dset.BitsStored = 16
        dset.HighBit = 15
    elif (data.dtype == "uint32") or (data.dtype == "int") or (data.dtype == "float32"):
        dset.BitsAllocated = 32
        dset.BitsStored = 32
        dset.HighBit = 31
    elif data.dtype == "float64":
        dset.BitsAllocated = 64
        dset.BitsStored = 64
        dset.HighBit = 63
    else:
        print("Unsupported data type: ", data.dtype)

    dset.SeriesNumber = head.image_series_index
    dset.InstanceNumber = head.image_index

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
    dset.ImageType[2] = IMTYPE_MAPS[head.image_type]
    dset.PixelSpacing = [
        float(head.field_of_view[0]) / data.shape[2],
        float(head.field_of_view[1]) / data.shape[3],
    ]
    dset.SliceThickness = head.field_of_view[2]
    dset.ImagePositionPatient = [
        head.position[0],
        head.position[1],
        head.position[2],
    ]
    dset.ImageOrientationPatient = [
        head.read_dir[0],
        head.read_dir[1],
        head.read_dir[2],
        head.phase_dir[0],
        head.phase_dir[1],
        head.phase_dir[2],
    ]

    time_sec = head.acquisition_time_stamp / 1000 / 2.5
    hour = int(np.floor(time_sec / 3600))
    min = int(np.floor((time_sec - hour * 3600) / 60))
    sec = time_sec - hour * 3600 - min * 60
    dset.AcquisitionTime = "%02.0f%02.0f%09.6f" % (hour, min, sec)
    dset.TriggerTime = head.physiology_time_stamp[0] / 2.5

    # ----- Update DICOM header from MRD Image MetaAttributes -----
    if meta.get("SeriesDescription") is not None:
        dset.SeriesDescription = meta["SeriesDescription"]

    if meta.get("SeriesDescriptionAdditional") is not None:
        dset.SeriesDescription = (
            dset.SeriesDescription + meta["SeriesDescriptionAdditional"]
        )

    if meta.get("ImageComment") is not None:
        dset.ImageComment = "_".join(meta["ImageComment"])

    if meta.get("ImageType") is not None:
        dset.ImageType = meta["ImageType"]

    if (meta.get("ImageRowDir") is not None) and (
        meta.get("ImageColumnDir") is not None
    ):
        dset.ImageOrientationPatient = [
            float(meta["ImageRowDir"][0]),
            float(meta["ImageRowDir"][1]),
            float(meta["ImageRowDir"][2]),
            float(meta["ImageColumnDir"][0]),
            float(meta["ImageColumnDir"][1]),
            float(meta["ImageColumnDir"][2]),
        ]

    if meta.get("RescaleIntercept") is not None:
        dset.RescaleIntercept = meta["RescaleIntercept"]

    if meta.get("RescaleSlope") is not None:
        dset.RescaleSlope = meta["RescaleSlope"]

    if meta.get("WindowCenter") is not None:
        dset.WindowCenter = meta["WindowCenter"]

    if meta.get("WindowWidth") is not None:
        dset.WindowWidth = meta["WindowWidth"]

    # setting sequence parameters
    if mrdhead.sequence_parameters is not None:
        if mrdhead.sequence_parameters.flip_angle_deg:
            dset.FlipAngle = mrdhead.sequence_parameters.flip_angle_deg
        if mrdhead.sequence_parameters.t_r:
            dset.RepetitionTime = mrdhead.sequence_parameters.t_r
        if mrdhead.sequence_parameters.t_e:
            dset.EchoTime = mrdhead.sequence_parameters.t_e
        if mrdhead.sequence_parameters.t_i:
            dset.InversionTime = mrdhead.sequence_parameters.t_i

    # setting spacing
    fov_z = mrdhead.encoding[-1].encoded_space.field_of_view_mm.z
    nz = mrdhead.encoding[-1].encoded_space.matrix_size.z
    dset.SpacingBetweenSlices = fov_z / nz

    return dset
