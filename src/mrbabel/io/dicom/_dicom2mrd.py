"""DICOM to MRD Conversion Utilities."""

__all__ = [
    "DEFAULTS",
    "IMTYPE_MAPS",
    "VENC_DIR_MAP",
    "read_dicom_header",
    "read_dicom_images",
]

import base64
import re
import warnings

import mrd
import numpy as np
import pydicom

# Defaults for input arguments
DEFAULTS = {
    "out_group": "dataset",
}

# Lookup table for image types, configurable by vendor
IMTYPE_MAPS = {
    "default": {
        "M": mrd.ImageType.MAGNITUDE,
        "P": mrd.ImageType.PHASE,
        "R": mrd.ImageType.REAL,
        "I": mrd.ImageType.IMAG,
    },
    "GE": {
        0: mrd.ImageType.MAGNITUDE,
        1: mrd.ImageType.PHASE,
        2: mrd.ImageType.REAL,
        3: mrd.ImageType.IMAG,
    },
}

# Lookup table between DICOM and Siemens flow directions
VENC_DIR_MAP = {
    "rl": "FLOW_DIR_R_TO_L",
    "lr": "FLOW_DIR_L_TO_R",
    "ap": "FLOW_DIR_A_TO_P",
    "pa": "FLOW_DIR_P_TO_A",
    "fh": "FLOW_DIR_F_TO_H",
    "hf": "FLOW_DIR_H_TO_F",
    "in": "FLOW_DIR_TP_IN",
    "out": "FLOW_DIR_TP_OUT",
}


def read_dicom_header(dset: pydicom.Dataset) -> mrd.Header:
    """Create MRD Header from a DICOM file."""
    mrdhead = mrd.Header()

    # fill patient information
    mrdhead.subject_information = mrd.SubjectInformationType()
    mrdhead.subject_information.patient_name = dset.PatientName
    mrdhead.subject_information.weight_kg = dset.PatientWeight
    try:
        mrdhead.subject_information.height_m = dset.PatientHeight
    except Exception:
        pass
    mrdhead.subject_information.patient_id = dset.PatientID
    mrdhead.subject_information.patient_birthdate = dset.PatientBirthDate
    mrdhead.subject_information.patient_gender = dset.PatientSex

    # fill study information
    mrdhead.study_information = mrd.StudyInformationType()
    mrdhead.study_information.study_date = dset.StudyDate
    mrdhead.study_information.study_time = dset.StudyTime
    try:
        mrdhead.study_information.study_id = dset.StudyID
    except Exception:
        pass
    mrdhead.study_information.study_description = dset.StudyDescription
    mrdhead.study_information.study_instance_uid = dset.StudyInstanceUID

    # fill measurement information
    mrdhead.measurement_information = mrd.MeasurementInformationType()
    mrdhead.measurement_information.measurement_id = dset.SeriesInstanceUID
    mrdhead.measurement_information.patient_position = dset.PatientPosition
    try:
        mrdhead.measurement_information.protocol_name = dset.SeriesDescription
    except Exception:
        pass
    mrdhead.measurement_information.frame_of_reference_uid = dset.FrameOfReferenceUID

    # fill acquisition system information
    mrdhead.acquisition_system_information = mrd.AcquisitionSystemInformationType()
    mrdhead.acquisition_system_information.system_vendor = dset.Manufacturer
    mrdhead.acquisition_system_information.system_model = dset.ManufacturerModelName
    mrdhead.acquisition_system_information.system_field_strength_t = float(
        dset.MagneticFieldStrength
    )
    try:
        mrdhead.acquisition_system_information.institution_name = dset.InstitutionName
    except Exception:
        mrdhead.acquisition_system_information.institution_name = "Virtual"
    try:
        mrdhead.acquisition_system_information.station_name = dset.StationName
    except Exception:
        pass

    # fill experimental condition
    mrdhead.experimental_conditions.h1resonance_frequency_hz = int(
        dset.MagneticFieldStrength * 4258e4
    )

    # fill encoding space
    enc = mrd.EncodingType()
    enc.trajectory = mrd.Trajectory.CARTESIAN
    enc.encoded_space.matrix_size.x = dset.Columns
    enc.encoded_space.matrix_size.y = dset.Rows
    enc.encoded_space.matrix_size.z = 1
    enc.encoded_space.field_of_view_mm = mrd.FieldOfViewMm()

    if dset.SOPClassUID.name == "Enhanced MR Image Storage":
        slice_thickness = float(
            dset.PerFrameFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .SliceThickness
        )
        try:
            slice_spacing = float(
                dset.PerFrameFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
                .SpacingBetweenSlices
            )
        except Exception:
            warnings.warn(
                "Slice thickness and spacing info not found; assuming contiguous"
                " slices!",
                UserWarning,
            )
            slice_spacing = slice_thickness

        enc.encoded_space.field_of_view_mm.x = (
            dset.PerFrameFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing[0]
            * dset.Rows
        )
        enc.encoded_space.field_of_view_mm.y = (
            dset.PerFrameFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing[1]
            * dset.Columns
        )
        enc.encoded_space.field_of_view_mm.z = float(slice_spacing)
    else:
        slice_thickness = float(dset["SliceThickness"].value)
        if "SpacingBetweenSlices" in dset:
            slice_spacing = float(dset["SpacingBetweenSlices"].value)
        else:
            warnings.warn(
                "Slice thickness and spacing info not found; assuming contiguous"
                " slices!",
                UserWarning,
            )
            slice_spacing = slice_thickness

        enc.encoded_space.field_of_view_mm.x = dset.PixelSpacing[0] * dset.Rows
        enc.encoded_space.field_of_view_mm.y = dset.PixelSpacing[1] * dset.Columns
        enc.encoded_space.field_of_view_mm.z = slice_spacing

    # fill recon space
    enc.recon_space = enc.encoded_space

    # fill encoding limit
    enc.encoding_limits.kspace_encoding_step_0 = mrd.LimitType()
    enc.encoding_limits.kspace_encoding_step_0.maximum = int(dset.Columns) - 1
    enc.encoding_limits.kspace_encoding_step_0.center = int(dset.Columns) // 2

    enc.encoding_limits.kspace_encoding_step_1 = mrd.LimitType()
    enc.encoding_limits.kspace_encoding_step_1.maximum = int(dset.Rows) - 1
    enc.encoding_limits.kspace_encoding_step_1.center = int(dset.Rows) // 2

    enc.encoding_limits.slice = mrd.LimitType()
    enc.encoding_limits.contrast = mrd.LimitType()

    enc.parallel_imaging = mrd.ParallelImagingType()
    if dset.SOPClassUID.name == "Enhanced MR Image Storage":
        enc.parallel_imaging.acceleration_factor.kspace_encoding_step_1 = (
            dset.SharedFunctionalGroupsSequence[0]
            .MRModifierSequence[0]
            .ParallelReductionFactorInPlane
        )
        enc.parallel_imaging.acceleration_factor.kspace_encoding_step_2 = (
            dset.SharedFunctionalGroupsSequence[0]
            .MRModifierSequence[0]
            .ParallelReductionFactorOutOfPlane
        )
    else:
        enc.parallel_imaging.acceleration_factor.kspace_encoding_step_1 = 1
        enc.parallel_imaging.acceleration_factor.kspace_encoding_step_2 = 1

    mrdhead.encoding.append(enc)
    mrdhead.sequence_parameters = mrd.SequenceParametersType()

    mrdhead.user_parameters = mrd.UserParametersType()

    # assign
    slice_thickness = mrd.UserParameterDoubleType(
        name="SliceThickness", value=slice_thickness
    )
    mrdhead.user_parameters.user_parameter_double.append(slice_thickness)
    slice_spacing = mrd.UserParameterDoubleType(
        name="SpacingBetweenSlices", value=slice_spacing
    )
    mrdhead.user_parameters.user_parameter_double.append(slice_spacing)

    return mrdhead


def read_dicom_images(
    dsets: list[pydicom.Dataset], mrdhead: mrd.Header
) -> list[mrd.Acquisition]:
    """Create list of MRD Acquisitions from a DICOM file."""
    images = []

    # Sort images by instance number, as they may be read out of order
    def get_instance_number(item):
        return item.InstanceNumber

    dsets = sorted(dsets, key=get_instance_number)

    # Build a list of unique Instance Numbers
    unique_series_numbers = np.unique([int(dset.SeriesNumber) for dset in dsets])

    # Build a list of unique SliceLocation for slice counter
    unique_slice_locations, slice_idx = _get_unique_slice_locations(dsets)

    # Build list of unique FA
    flip_angles = np.asarray([float(dset.FlipAngle) for dset in dsets])

    # Build list of unique TE
    echo_times = np.asarray([float(dset.EchoTime) for dset in dsets])

    # Build list of unique TR
    repetition_times = np.asarray([float(dset.RepetitionTime) for dset in dsets])

    # Build list of unique TI
    try:
        inversion_times = np.asarray([float(dset.InversionTime) for dset in dsets])
    except Exception:
        inversion_times = -1 * np.ones_like(repetition_times)

    # Build list of contrasts
    contrasts = np.stack(
        (inversion_times, echo_times, repetition_times, flip_angles), axis=1
    )
    ncontrasts = contrasts.shape[0]

    # Get unique contrast and indexes and update mrd header
    unique_contrasts, contrast_idx = _get_unique_contrasts(contrasts)
    mrdhead.sequence_parameters.flip_angle_deg = np.unique(unique_contrasts[3])
    mrdhead.sequence_parameters.t_r = np.unique(unique_contrasts[2])
    mrdhead.sequence_parameters.t_e = np.unique(unique_contrasts[1])
    if "InversionTime" in dsets[0]:
        mrdhead.sequence_parameters.t_i = np.unique(unique_contrasts[0])

    # Get number of slices and update mrd header
    nslices = len(unique_slice_locations)
    mrdhead.encoding[0].encoded_space.field_of_view_mm.z *= nslices
    mrdhead.encoding[0].encoded_space.matrix_size.z *= nslices

    mrdhead.encoding[0].recon_space.field_of_view_mm.z *= nslices
    mrdhead.encoding[0].recon_space.matrix_size.z *= nslices

    # Get vendor as image type map depends on this
    vendor = dsets[0].Manufacturer

    def get_image_type(item):
        if "GE" in vendor.upper():
            try:
                return IMTYPE_MAPS["GE"][item[0x0043, 0x102F].value]
            except Exception:
                pass
        return IMTYPE_MAPS["default"][item.ImageType[2]]

    # get limits
    mrdhead.encoding[0].encoding_limits.slice.maximum = nslices - 1
    mrdhead.encoding[0].encoding_limits.slice.center = nslices // 2

    mrdhead.encoding[0].encoding_limits.contrast.maximum = ncontrasts - 1
    mrdhead.encoding[0].encoding_limits.contrast.center = ncontrasts // 2

    # Loop over DICOM dataset and build Image
    for n in range(len(dsets)):
        dset = dsets[n]

        # Get image type
        try:
            image_type = get_image_type(dset)
        except:
            image_type = mrd.ImageType.MAGNITUDE

        # Initialize current image header
        head = mrd.ImageHeader(image_type=image_type)

        # Fill resolution
        head.field_of_view = (
            dset.PixelSpacing[0] * dset.Rows,
            dset.PixelSpacing[1] * dset.Columns,
            dset.SliceThickness,
        )

        # Fill position and orientation
        head.position = tuple(np.stack(dset.ImagePositionPatient))
        head.read_dir = tuple(np.stack(dset.ImageOrientationPatient[0:3]))
        head.phase_dir = tuple(np.stack(dset.ImageOrientationPatient[3:7]))
        head.slice_dir = tuple(
            np.cross(
                np.stack(dset.ImageOrientationPatient[0:3]),
                np.stack(dset.ImageOrientationPatient[3:7]),
            )
        )

        # Fill acquisition timestamp
        acquisition_time = "".join(dset.AcquisitionTime.split(":"))
        head.acquisition_time_stamp = round(
            (
                int(acquisition_time[0:2]) * 3600
                + int(acquisition_time[2:4]) * 60
                + int(acquisition_time[4:6])
                + float(acquisition_time[6:])
            )
            * 1000
            / 2.5
        )

        # Fill trigger
        # try:
        #     head.physiology_time_stamp[0] = round(
        #         int(dset.TriggerTime / 2.5)
        #     )
        # except Exception:
        #     pass

        # Fill table position
        try:
            ImaAbsTablePosition = dset.get_private_item(
                0x0019, 0x13, "SIEMENS MR HEADER"
            ).value
            head.patient_table_position = (
                float(ImaAbsTablePosition[0]),
                float(ImaAbsTablePosition[1]),
                float(ImaAbsTablePosition[2]),
            )
        except Exception:
            pass

        # Label data
        head.image_series_index = unique_series_numbers.tolist().index(
            dset.SeriesNumber
        )
        head.image_index = dset.get("InstanceNumber", 0)
        head.slice = slice_idx[n]
        # head.phase = trigger_idx[n]
        head.contrast = contrast_idx[n]

        # Fill current Meta values
        meta = {}

        try:
            res = re.search(r"(?<=_v).*$", dset.SequenceName)
            venc = re.search(r"^\d+", dset.group(0))
            dir = re.search(r"(?<=\d)[^\d]*$", res.group(0))

            meta["FlowVelocity"] = float(venc.group(0))
            meta["FlowDirDisplay"] = VENC_DIR_MAP[dir.group(0)]
        except Exception:
            pass

        try:
            meta["ImageComments"] = dset.ImageComments
        except Exception:
            pass

        try:
            meta["SeriesDescription"] = dset.SeriesDescription
        except Exception:
            pass

        # Remove pixel data from pydicom class
        try:
            data = dset.pixel_array.copy().astype(np.float32)
            del dset["PixelData"]
        except Exception:
            try:
                data = (
                    np.frombuffer(dset.FloatPixelData, dtype=np.float32)
                    .copy()
                    .reshape(dset.Rows, dset.Columns)
                )
                del dset["FloatPixelData"]
            except Exception:
                data = (
                    np.frombuffer(dset.DoubleFloatPixelData, dtype=np.float64)
                    .copy()
                    .astype(np.float32)
                    .reshape(dset.Rows)
                )
                del dset["DoubleFloatPixelData"]

        # Store the complete base64, json-formatted DICOM header so that non-MRD fields can be
        # recapitulated when generating DICOMs from MRD images
        meta["dicom_json"] = base64.b64encode(dset.to_json().encode("utf-8")).decode(
            "utf-8"
        )

        images.append(mrd.Image(head=head, data=data, meta=meta))

    return images, mrdhead


# %% local utils
def _get_unique_contrasts(constrasts):
    """Return ndarray of unique contrasts and contrast index for each dataset in dsets."""
    unique_contrasts = np.unique(constrasts, axis=0)

    # get indexes
    contrast_idx = np.zeros(constrasts.shape[0], dtype=int)

    for n in range(unique_contrasts.shape[0]):
        contrast_idx[(constrasts == unique_contrasts[n]).all(axis=-1)] = n

    return unique_contrasts.T, contrast_idx


def _get_unique_slice_locations(dsets):
    """Return array of unique slice locations and slice location index for each dataset in dsets."""
    # get unique slice locations
    slice_locs = _get_relative_slice_position(dsets).round(decimals=4)
    unique_slice_locs = np.unique(slice_locs)

    # get indexes
    slice_idx = np.zeros(slice_locs.shape[0], dtype=int)

    for n in range(len(unique_slice_locs)):
        slice_idx[slice_locs == unique_slice_locs[n]] = n

    return unique_slice_locs, slice_idx


# def _get_unique_trigger_times(dsets):
#     """Return array of unique trigger times and trigger time index for each dataset in dsets."""
#     trigger_times = np.asarray([float(dset.TriggerTime) for dset in dsets])
#     unique_trigger_times = np.unique(trigger_times)

#     # get indexes
#     trigger_idx = np.zeros(trigger_times.shape[0], dtype=int)

#     for n in range(unique_trigger_times.shape[0]):
#         trigger_idx[trigger_times == unique_trigger_times[n]] = n

#     return unique_trigger_times, trigger_idx


def _get_image_orientation(dsets):
    """Return image orientation matrix."""
    return np.array(dsets[0].ImageOrientationPatient).reshape(2, 3)


def _get_plane_normal(dsets):
    """Return array of normal to imaging plane, as the cross product between x and y plane versors."""
    x, y = _get_image_orientation(dsets)
    return np.cross(x, y)


def _get_position(dsets):
    """Return matrix of image position of size (3, nslices)."""
    return np.stack([dset.ImagePositionPatient for dset in dsets], axis=1)


def _get_relative_slice_position(dsets):
    """Return array of slice coordinates along the normal to imaging plane."""
    z = _get_plane_normal(dsets)
    position = _get_position(dsets)
    return z @ position
