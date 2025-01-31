"""DICOM to MRD Conversion Utilities."""

__all__ = [
    "DEFAULTS",
    "IMTYPE_MAPS",
    "VENC_DIR_MAP",
    "read_dicom_header",
    "read_dicom_images",
]

import base64
import copy
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


def _convert_patient_position(PatientPosition):
    return mrd.PatientPosition[PatientPosition[0] + "_" + PatientPosition[1:]]


def read_dicom_header(dset: pydicom.Dataset) -> mrd.Header:
    """Create MRD Header from a DICOM file."""
    head = mrd.Header(version=2.0)

    # fill patient information
    head.subject_information = mrd.SubjectInformationType()
    try:
        head.subject_information.patient_name = dset.PatientName
    except Exception:
        pass
    head.subject_information.patient_weight_kg = dset.PatientWeight
    try:
        head.subject_information.patient_height_m = dset.PatientHeight
    except Exception:
        pass
    head.subject_information.patient_id = dset.PatientID
    try:
        head.subject_information.patient_birthdate = dset.PatientBirthDate
    except Exception:
        pass
    try:
        head.subject_information.patient_gender = mrd.PatientGender[dset.PatientSex]
    except Exception:
        head.subject_information.patient_gender = mrd.PatientGender["O"]

    # fill study information
    head.study_information = mrd.StudyInformationType()
    head.study_information.study_date = dset.StudyDate
    head.study_information.study_time = dset.StudyTime
    try:
        head.study_information.study_id = dset.StudyID
    except Exception:
        pass
    try:
        head.study_information.study_description = dset.StudyDescription
    except Exception:
        pass
    head.study_information.study_instance_uid = dset.StudyInstanceUID

    # fill measurement information
    head.measurement_information = mrd.MeasurementInformationType()
    head.measurement_information.measurement_id = dset.SeriesInstanceUID
    head.measurement_information.patient_position = _convert_patient_position(
        dset.PatientPosition
    )
    try:
        head.measurement_information.protocol_name = dset.ProtocolName
    except Exception:
        pass
    try:
        head.measurement_information.series_description = dset.SeriesDescription
    except Exception:
        pass
    try:
        head.measurement_information.series_date = dset.SeriesDate
    except Exception:
        pass
    try:
        head.measurement_information.series_time = dset.SeriesTime
    except Exception:
        pass
    try:
        head.measurement_information.initial_series_number = dset.SeriesNumber
    except Exception:
        pass
    head.measurement_information.frame_of_reference_uid = dset.FrameOfReferenceUID

    # fill acquisition system information
    head.acquisition_system_information = mrd.AcquisitionSystemInformationType()
    head.acquisition_system_information.system_vendor = dset.Manufacturer
    head.acquisition_system_information.system_model = dset.ManufacturerModelName
    head.acquisition_system_information.system_field_strength_t = float(
        dset.MagneticFieldStrength
    )
    try:
        head.acquisition_system_information.institution_name = dset.InstitutionName
    except Exception:
        head.acquisition_system_information.institution_name = "Virtual"
    try:
        head.acquisition_system_information.station_name = dset.StationName
    except Exception:
        pass

    # fill experimental condition
    head.experimental_conditions.h1resonance_frequency_hz = int(
        dset.MagneticFieldStrength * 4258e4
    )

    # fill encoding space
    encoding = mrd.EncodingType()
    encoding.trajectory = mrd.Trajectory.CARTESIAN
    encoding.encoded_space.matrix_size.x = dset.Columns
    encoding.encoded_space.matrix_size.y = dset.Rows
    encoding.encoded_space.matrix_size.z = 1
    encoding.encoded_space.field_of_view_mm = mrd.FieldOfViewMm()

    if (
        hasattr(dset, "SOPClassUID")
        and dset.SOPClassUID.name == "Enhanced MR Image Storage"
    ):
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

        encoding.encoded_space.field_of_view_mm.x = (
            dset.PerFrameFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing[0]
            * dset.Rows
        )
        encoding.encoded_space.field_of_view_mm.y = (
            dset.PerFrameFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
            .PixelSpacing[1]
            * dset.Columns
        )
        encoding.encoded_space.field_of_view_mm.z = float(slice_spacing)
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

        encoding.encoded_space.field_of_view_mm.x = dset.PixelSpacing[0] * dset.Rows
        encoding.encoded_space.field_of_view_mm.y = dset.PixelSpacing[1] * dset.Columns
        encoding.encoded_space.field_of_view_mm.z = slice_spacing

    # fill recon space
    encoding.recon_space = copy.deepcopy(encoding.encoded_space)

    # fill encoding limit
    encoding.encoding_limits.kspace_encoding_step_0 = mrd.LimitType()
    encoding.encoding_limits.kspace_encoding_step_0.maximum = int(dset.Columns) - 1
    encoding.encoding_limits.kspace_encoding_step_0.center = int(dset.Columns) // 2

    encoding.encoding_limits.kspace_encoding_step_1 = mrd.LimitType()
    encoding.encoding_limits.kspace_encoding_step_1.maximum = int(dset.Rows) - 1
    encoding.encoding_limits.kspace_encoding_step_1.center = int(dset.Rows) // 2

    encoding.encoding_limits.slice = mrd.LimitType()
    encoding.encoding_limits.contrast = mrd.LimitType()

    encoding.parallel_imaging = mrd.ParallelImagingType()
    if (
        hasattr(dset, "SOPClassUID")
        and dset.SOPClassUID.name == "Enhanced MR Image Storage"
    ):
        encoding.parallel_imaging.acceleration_factor.kspace_encoding_step_1 = (
            dset.SharedFunctionalGroupsSequence[0]
            .MRModifierSequence[0]
            .ParallelReductionFactorInPlane
        )
        encoding.parallel_imaging.acceleration_factor.kspace_encoding_step_2 = (
            dset.SharedFunctionalGroupsSequence[0]
            .MRModifierSequence[0]
            .ParallelReductionFactorOutOfPlane
        )
    else:
        encoding.parallel_imaging.acceleration_factor.kspace_encoding_step_1 = 1
        encoding.parallel_imaging.acceleration_factor.kspace_encoding_step_2 = 1

    head.encoding.append(encoding)
    head.sequence_parameters = mrd.SequenceParametersType()
    head.sequence_parameters.flip_angle_deg = [float(dset.FlipAngle)]
    head.sequence_parameters.t_r = [float(dset.RepetitionTime)]
    head.sequence_parameters.t_e = [float(dset.EchoTime)]
    try:
        head.sequence_parameters.t_i = [float(dset.InversionTime)]
    except Exception:
        pass

    head.user_parameters = mrd.UserParametersType()

    # Slice Thickness and Spacing
    slice_thickness = mrd.UserParameterDoubleType(
        name="SliceThickness", value=slice_thickness
    )
    head.user_parameters.user_parameter_double.append(slice_thickness)
    slice_spacing = mrd.UserParameterDoubleType(
        name="SpacingBetweenSlices", value=slice_spacing
    )
    head.user_parameters.user_parameter_double.append(slice_spacing)

    # Imaging Mode
    imode = mrd.UserParameterStringType(
        name="ImagingMode", value=dset.MRAcquisitionType
    )
    head.user_parameters.user_parameter_string.append(imode)

    return head


def read_dicom_images(
    dsets: list[pydicom.Dataset], head: mrd.Header
) -> list[mrd.Acquisition]:
    """Create list of MRD Acquisitions from a DICOM file."""
    images = []

    # Sort images by instance number, as they may be read out of order
    def get_instance_number(item):
        try:
            return item.InstanceNumber
        except Exception:
            return 0

    dsets = sorted(dsets, key=get_instance_number)

    # Build a list of unique Instance Numbers
    # unique_series_numbers = np.unique([int(dset.SeriesNumber) for dset in dsets])

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
    n_contrasts = contrasts.shape[0]

    # Get unique contrast and indexes and update mrd header
    unique_contrasts, contrast_idx = _get_unique_contrasts(contrasts)
    head.sequence_parameters.flip_angle_deg = np.unique(unique_contrasts[3])
    head.sequence_parameters.t_r = np.unique(unique_contrasts[2])
    head.sequence_parameters.t_e = np.unique(unique_contrasts[1])
    if "InversionTime" in dsets[0]:
        head.sequence_parameters.t_i = np.unique(unique_contrasts[0])

    # Get number of slices and update mrd header
    n_slices = len(unique_slice_locations)
    head.encoding[-1].encoded_space.field_of_view_mm.z *= n_slices
    head.encoding[-1].encoded_space.matrix_size.z *= n_slices

    head.encoding[-1].recon_space.field_of_view_mm.z *= n_slices
    head.encoding[-1].recon_space.matrix_size.z *= n_slices

    # Get vendor as image type map depends on this
    vendor = dsets[0].Manufacturer

    def get_image_type(item):
        if "GE" in vendor.upper():
            try:
                return IMTYPE_MAPS["GE"][item[0x0043, 0x102F].value]
            except Exception:
                pass
        return IMTYPE_MAPS["default"][item.ImageType[2][0]]

    # get limits
    head.encoding[-1].encoding_limits.slice.maximum = n_slices - 1
    head.encoding[-1].encoding_limits.slice.center = n_slices // 2

    head.encoding[-1].encoding_limits.contrast.maximum = n_contrasts - 1
    head.encoding[-1].encoding_limits.contrast.center = n_contrasts // 2

    # Loop over DICOM dataset and build Image
    for n in range(len(dsets)):
        dset = dsets[n]

        # Get image type
        try:
            image_type = get_image_type(dset)
        except Exception:
            image_type = mrd.ImageType.MAGNITUDE

        # Initialize current image header
        image_head = mrd.ImageHeader(image_type=image_type)
        
        # Flags
        defs = mrd.ImageFlags
        if slice_idx[n] == 0:
            image_head.flags = defs.FIRST_IN_SLICE
        if slice_idx[n] == n_slices - 1:
            image_head.flags = defs.LAST_IN_SLICE
        if contrast_idx[n] == 0:
            image_head.flags = defs.FIRST_IN_CONTRAST
        if contrast_idx[n] == n_contrasts - 1:
            image_head.flags = defs.LAST_IN_CONTRAST

        # Fill resolution
        image_head.field_of_view = (
            dset.PixelSpacing[0] * dset.Rows,
            dset.PixelSpacing[1] * dset.Columns,
            dset.SliceThickness,
        )

        # Fill position and orientation
        image_head.position = tuple(np.stack(dset.ImagePositionPatient))
        image_head.line_dir = tuple(np.stack(dset.ImageOrientationPatient[0:3]))
        image_head.col_dir = tuple(np.stack(dset.ImageOrientationPatient[3:7]))
        image_head.slice_dir = tuple(
            np.cross(
                np.stack(dset.ImageOrientationPatient[0:3]),
                np.stack(dset.ImageOrientationPatient[3:7]),
            )
        )

        # Fill acquisition timestamp
        try:
            acquisition_time = "".join(dset.AcquisitionTime.split(":"))
            image_head.acquisition_time_stamp = round(
                (
                    int(acquisition_time[0:2]) * 3600
                    + int(acquisition_time[2:4]) * 60
                    + int(acquisition_time[4:6])
                    + float(acquisition_time[6:])
                )
                * 1000
                / 2.5
            )
        except Exception:
            pass

        # Fill trigger
        # try:
        #     image_head.physiology_time_stamp[0] = round(
        #         int(dset.TriggerTime / 2.5)
        #     )
        # except Exception:
        #     pass

        # Fill table position
        try:
            ImaAbsTablePosition = dset.get_private_item(
                0x0019, 0x13, "SIEMENS MR HEADER"
            ).value
            image_head.patient_table_position = (
                float(ImaAbsTablePosition[0]),
                float(ImaAbsTablePosition[1]),
                float(ImaAbsTablePosition[2]),
            )
        except Exception:
            pass

        # Label data
        image_head.image_series_index = int(dset.SeriesNumber)
        image_head.image_index = int(dset.get("InstanceNumber", 0))
        image_head.slice = slice_idx[n]
        # image_head.phase = trigger_idx[n]
        image_head.contrast = contrast_idx[n]

        # Fill current image_meta values
        image_meta = mrd.ImageMeta()

        try:
            res = re.search(r"(?<=_v).*$", dset.SequenceName)
            venc = re.search(r"^\d+", dset.group(0))
            dir = re.search(r"(?<=\d)[^\d]*$", res.group(0))

            image_meta["FlowVelocity"] = float(venc.group(0))
            image_meta["FlowDirDisplay"] = VENC_DIR_MAP[dir.group(0)]
        except Exception:
            pass

        try:
            image_meta["ImageComments"] = dset.ImageComments
        except Exception:
            pass

        try:
            image_meta["SeriesDescription"] = dset.SeriesDescription
        except Exception:
            pass

        # Remove pixel data from pydicom class
        try:
            image_data = dset.pixel_array.copy().astype(np.float32)
            del dset["PixelData"]
        except Exception:
            try:
                image_data = (
                    np.frombuffer(dset.FloatPixelData, dtype=np.float32)
                    .copy()
                    .reshape(dset.Rows, dset.Columns)
                )
                del dset["FloatPixelData"]
            except Exception:
                try:
                    image_data = (
                        np.frombuffer(dset.DoubleFloatPixelData, dtype=np.float64)
                        .copy()
                        .astype(np.float32)
                        .reshape(dset.Rows)
                    )
                    del dset["DoubleFloatPixelData"]
                except Exception:
                    image_data = None

        # Store the complete base64, json-formatted DICOM header so that non-MRD fields can be
        # recapitulated when generating DICOMs from MRD images
        image_meta["DicomJson"] = base64.b64encode(
            dset.to_json().encode("utf-8")
        ).decode("utf-8")

        images.append(mrd.Image(head=image_head, data=image_data, meta=image_meta))

    return images, head


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
    """Return matrix of image position of size (3, n_slices)."""
    return np.stack([dset.ImagePositionPatient for dset in dsets], axis=1)


def _get_relative_slice_position(dsets):
    """Return array of slice coordinates along the normal to imaging plane."""
    z = _get_plane_normal(dsets)
    position = _get_position(dsets)
    return z @ position
