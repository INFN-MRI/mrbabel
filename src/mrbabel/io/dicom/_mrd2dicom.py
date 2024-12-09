"""MRD to DICOM Conversion Utilities."""

__all__ = [
    "DEFAULTS",
    "IMTYPE_MAPS",
    "VENC_DIR_MAP",
    "dump_dicom_images",
]

import base64
import re

import mrd
import numpy as np
import pydicom

# Defaults for input arguments
DEFAULTS = {
    "out_group": "dataset",
}

# Lookup table for image types, configurable by vendor
IMTYPE_MAPS = {
    mrd.ImageType.MAGNITUDE: {"default": "M", "GE": 0},
    mrd.ImageType.PHASE: {"default": "P", "GE": 1},
    mrd.ImageType.REAL: {"default": "R", "GE": 2},
    mrd.ImageType.IMAG: {"default": "I", "GE": 3},
}

# Lookup table between DICOM and Siemens flow directions
VENC_DIR_MAP = {
    "FLOW_DIR_R_TO_L": "rl",
    "FLOW_DIR_L_TO_R": "lr",
    "FLOW_DIR_A_TO_P": "ap",
    "FLOW_DIR_P_TO_A": "pa",
    "FLOW_DIR_F_TO_H": "fh",
    "FLOW_DIR_H_TO_F": "hf",
    "FLOW_DIR_TP_IN": "in",
    "FLOW_DIR_TP_OUT": "out",
}


def dump_dicom_images(
    dsets: list[pydicom.Dataset], mrdhead: mrd.Header
) -> list[mrd.Acquisition]: ...


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
    try:
        if mrdhead.measurement_information is None:
            pass
        else:
            if mrdhead.measurementInformation.measurementID is not None:
                dset.SeriesInstanceUID = mrdhead.measurementInformation.measurementID
            if mrdhead.measurementInformation.patientPosition is not None:
                dset.PatientPosition = (
                    mrdhead.measurementInformation.patientPosition.name
                )
            if mrdhead.measurementInformation.protocolName is not None:
                dset.SeriesDescription = mrdhead.measurementInformation.protocolName
            if mrdhead.measurementInformation.frameOfReferenceUID is not None:
                dset.FrameOfReferenceUID = (
                    mrdhead.measurementInformation.frameOfReferenceUID
                )

            # print("---------- New -------------------------")
            # print("SeriesInstanceUID  : %s" % dset.SeriesInstanceUID   )
            # print("PatientPosition    : %s" % dset.PatientPosition     )
            # print("SeriesDescription  : %s" % dset.SeriesDescription   )
            # print("FrameOfReferenceUID: %s" % dset.FrameOfReferenceUID )
    except:
        print(
            "Error setting header information from MRD header's measurementInformation section"
        )

    try:
        if mrdhead.acquisitionSystemInformation is None:
            pass
        else:
            # print("---------- Old -------------------------")
            # print("mrdhead.acquisitionSystemInformation.systemVendor         : %s" % mrdhead.acquisitionSystemInformation.systemVendor          )
            # print("mrdhead.acquisitionSystemInformation.systemModel          : %s" % mrdhead.acquisitionSystemInformation.systemModel           )
            # print("mrdhead.acquisitionSystemInformation.systemFieldStrength_T: %s" % mrdhead.acquisitionSystemInformation.systemFieldStrength_T )
            # print("mrdhead.acquisitionSystemInformation.institutionName      : %s" % mrdhead.acquisitionSystemInformation.institutionName       )
            # print("mrdhead.acquisitionSystemInformation.stationName          : %s" % mrdhead.acquisitionSystemInformation.stationName           )

            if mrdhead.acquisitionSystemInformation.systemVendor is not None:
                dset.Manufacturer = mrdhead.acquisitionSystemInformation.systemVendor
            if mrdhead.acquisitionSystemInformation.systemModel is not None:
                dset.ManufacturerModelName = (
                    mrdhead.acquisitionSystemInformation.systemModel
                )
            if mrdhead.acquisitionSystemInformation.systemFieldStrength_T is not None:
                dset.MagneticFieldStrength = (
                    mrdhead.acquisitionSystemInformation.systemFieldStrength_T
                )
            if mrdhead.acquisitionSystemInformation.institutionName is not None:
                dset.InstitutionName = (
                    mrdhead.acquisitionSystemInformation.institutionName
                )
            if mrdhead.acquisitionSystemInformation.stationName is not None:
                dset.StationName = mrdhead.acquisitionSystemInformation.stationName

            # print("---------- New -------------------------")
            # print("mrdhead.acquisitionSystemInformation.systemVendor         : %s" % mrdhead.acquisitionSystemInformation.systemVendor          )
            # print("mrdhead.acquisitionSystemInformation.systemModel          : %s" % mrdhead.acquisitionSystemInformation.systemModel           )
            # print("mrdhead.acquisitionSystemInformation.systemFieldStrength_T: %s" % mrdhead.acquisitionSystemInformation.systemFieldStrength_T )
            # print("mrdhead.acquisitionSystemInformation.institutionName      : %s" % mrdhead.acquisitionSystemInformation.institutionName       )
            # print("mrdhead.acquisitionSystemInformation.stationName          : %s" % mrdhead.acquisitionSystemInformation.stationName           )
    except:
        print(
            "Error setting header information from MRD header's acquisitionSystemInformation section"
        )

    # Set mrdImg pixel data from MRD mrdImg
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

    dset.SeriesNumber = mrdImg.image_series_index
    dset.InstanceNumber = mrdImg.image_index

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
    dset.ImageType[2] = imtype_map[mrdImg.image_type]
    dset.PixelSpacing = [
        float(mrdImg.field_of_view[0]) / data.shape[2],
        float(mrdImg.field_of_view[1]) / data.shape[3],
    ]
    dset.SliceThickness = mrdImg.field_of_view[2]
    dset.ImagePositionPatient = [
        mrdImg.position[0],
        mrdImg.position[1],
        mrdImg.position[2],
    ]
    dset.ImageOrientationPatient = [
        mrdImg.read_dir[0],
        mrdImg.read_dir[1],
        mrdImg.read_dir[2],
        mrdImg.phase_dir[0],
        mrdImg.phase_dir[1],
        mrdImg.phase_dir[2],
    ]

    time_sec = mrdImg.acquisition_time_stamp / 1000 / 2.5
    hour = int(np.floor(time_sec / 3600))
    min = int(np.floor((time_sec - hour * 3600) / 60))
    sec = time_sec - hour * 3600 - min * 60
    dset.AcquisitionTime = "%02.0f%02.0f%09.6f" % (hour, min, sec)
    dset.TriggerTime = mrdImg.physiology_time_stamp[0] / 2.5

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

    if meta.get("EchoTime") is not None:
        dset.EchoTime = meta["EchoTime"]

    if meta.get("InversionTime") is not None:
        dset.InversionTime = meta["InversionTime"]
