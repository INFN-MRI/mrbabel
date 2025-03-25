"""GE to MRD Conversion Utilities."""

__all__ = [
    "read_gehc_header",
    "read_ismrmrd_acquisitions",
    "read_ismrmrd_acquisition",
    "read_ismrmrd_waveform",
]

from datetime import datetime, timedelta

import ismrmrd
import mrd

from types import SimpleNamespace


def read_gehc_header(hdr: SimpleNamespace, h: mrd.Header = None) -> mrd.Header:
    """Create MRD Header from a ISMRMRD Header."""
    if h is None:
        h = mrd.Header()

    if hasattr(hdr, "subjectInformation") and hdr.subjectInformation:
        h.subject_information = convert_subject_information(hdr.subjectInformation)
    if hasattr(hdr, "studyInformation") and hdr.studyInformation:
        h.study_information = convert_study_information(hdr.studyInformation)
    if hasattr(hdr, "measurementInformation") and hdr.measurementInformation:
        h.measurement_information = convert_measurement_information(
            hdr.measurementInformation
        )
    if (
        hasattr(hdr, "acquisitionSystemInformation")
        and hdr.acquisitionSystemInformation
    ):
        h.acquisition_system_information = convert_acquisition_system_information(
            hdr.acquisitionSystemInformation
        )

    h.experimental_conditions = convert_experimental_conditions(
        hdr.experimentalConditions
    )

    if hasattr(hdr, "encoding") and len(hdr.encoding) > 0:
        h.encoding = [convert_encoding(e) for e in hdr.encoding]
    else:
        raise RuntimeError("No encoding found in ISMRMRD header")

    if hasattr(hdr, "sequenceParameters") and hdr.sequenceParameters:
        h.sequence_parameters = convert_sequence_parameters(hdr.sequenceParameters)

    if hasattr(hdr, "userParameters") and hdr.userParameters:
        h.user_parameters = convert_user_parameters(hdr.userParameters)

    h.waveform_information = [read_ismrmrd_waveform(w) for w in hdr.waveformInformation]

    return h


def read_ismrmrd_acquisitions(
    acquisitions: list[ismrmrd.Acquisition],
) -> list[mrd.Acquisition]:
    """Create a lis tof MRD Acquisitions from a list of ISMRMRD Acquisitions."""
    return [read_ismrmrd_acquisition(acq) for acq in acquisitions]


def read_ismrmrd_acquisition(
    acq: ismrmrd.Acquisition, acquisition: mrd.Acquisition = None
) -> mrd.Acquisition:
    """Create MRD Acquisition from a ISMRMRD Acquisition."""
    if acquisition is None:
        acquisition = mrd.Acquisition()

    # Fill in the header fields
    acquisition.head.flags = acq.flags
    acquisition.head.idx = convert_encoding_counters(acq.idx)
    acquisition.head.measurement_uid = acq.measurement_uid
    acquisition.head.scan_counter = acq.scan_counter
    acquisition.head.acquisition_time_stamp = acq.acquisition_time_stamp
    acquisition.head.physiology_time_stamp = list(acq.physiology_time_stamp)
    for n in range(acq.active_channels):
        acquisition.head.channel_order.append(n)

    acquisition.head.discard_pre = acq.discard_pre
    acquisition.head.discard_post = acq.discard_post
    acquisition.head.center_sample = acq.center_sample
    acquisition.head.encoding_space_ref = acq.encoding_space_ref
    acquisition.head.sample_time_us = acq.sample_time_us

    acquisition.head.position = list(acq.position)
    acquisition.head.read_dir = list(acq.read_dir)
    acquisition.head.phase_dir = list(acq.phase_dir)
    acquisition.head.slice_dir = list(acq.slice_dir)
    acquisition.head.patient_table_position = list(acq.patient_table_position)

    acquisition.head.user_int.extend(list(acq.user_int))
    acquisition.head.user_float.extend(list(acq.user_float))

    # Resize the data structure (for example, using numpy arrays or lists)
    acquisition.data = acq.data

    # If trajectory dimensions are present, resize and fill the trajectory data
    if acq.trajectory_dimensions > 0:
        acquisition.trajectory = acq.traj

    return acquisition


def read_ismrmrd_waveform(
    wfm: ismrmrd.Waveform, waveform: mrd.Waveform = None
) -> mrd.Waveform:
    """Create MRD Waveform from a ISMRMRD Waveform."""
    if waveform is None:
        waveform = mrd.Waveform()
    waveform.flags = wfm.head.flags
    waveform.measurement_uid = wfm.head.measurement_uid
    waveform.scan_counter = wfm.head.scan_counter
    waveform.time_stamp = wfm.head.time_stamp
    waveform.sample_time_us = wfm.head.sample_time_us
    waveform.data = wfm.data

    return waveform


# %% subroutines
def string_to_date(date_str: str) -> timedelta:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    epoch = datetime(1970, 1, 1)
    return timedelta(days=(date_obj - epoch).days)


def time_to_string(time_str: str) -> timedelta:
    time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return timedelta(seconds=total_seconds)


def date_from_string(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Invalid date format")


def time_from_string(s):
    try:
        return datetime.strptime(s, "%H:%M:%S").time()
    except ValueError:
        raise ValueError("Invalid time format")


def convert_subject_information(subject_information, s=None):
    if s is None:
        s = mrd.SubjectInformationType()

    if subject_information.patientName:
        s.patient_name = subject_information.patientName

    if subject_information.patientWeight_kg:
        s.patient_weight_kg = subject_information.patientWeight_kg

    if subject_information.patientHeight_m:
        s.patient_height_m = subject_information.patientHeight_m

    if subject_information.patientID:
        s.patient_id = subject_information.patientID

    if subject_information.patientBirthdate:
        s.patient_birthdate = date_from_string(subject_information.patientBirthdate)

    if subject_information.patientGender:
        if subject_information.patientGender == "M":
            s.patient_gender = mrd.PatientGender.M
        elif subject_information.patientGender == "F":
            s.patient_gender = mrd.PatientGender.F
        elif subject_information.patientGender == "O":
            s.patient_gender = mrd.PatientGender.O
        else:
            raise ValueError("Unknown Gender")

    return s


def convert_study_information(study_information, s=None):
    if s is None:
        s = mrd.StudyInformationType()

    if (
        hasattr(study_information, "studyDate")
        and study_information.studyDate is not None
    ):
        s.study_date = date_from_string(study_information.studyDate)

    if (
        hasattr(study_information, "studyTime")
        and study_information.studyTime is not None
    ):
        s.study_time = time_from_string(study_information.studyTime)

    if hasattr(study_information, "studyID") and study_information.studyID is not None:
        s.study_id = study_information.studyID

    if (
        hasattr(study_information, "accessionNumber")
        and study_information.accessionNumber is not None
    ):
        s.accession_number = study_information.accessionNumber

    if (
        hasattr(study_information, "referringPhysicianName")
        and study_information.referringPhysicianName is not None
    ):
        s.referring_physician_name = study_information.referringPhysicianName

    if (
        hasattr(study_information, "studyDescription")
        and study_information.studyDescription is not None
    ):
        s.study_description = study_information.studyDescription

    if (
        hasattr(study_information, "studyInstanceUID")
        and study_information.studyInstanceUID is not None
    ):
        s.study_instance_uid = study_information.studyInstanceUID

    if (
        hasattr(study_information, "bodyPartExamined")
        and study_information.bodyPartExamined is not None
    ):
        s.body_part_examined = study_information.bodyPartExamined

    return s


def patient_position_from_string(s: str) -> str:
    position_map = {
        "HFP": mrd.PatientPosition.H_FP,
        "HFS": mrd.PatientPosition.H_FS,
        "HFDR": mrd.PatientPosition.H_FDR,
        "HFDL": mrd.PatientPosition.H_FDL,
        "FFP": mrd.PatientPosition.F_FP,
        "FFS": mrd.PatientPosition.F_FS,
        "FFDR": mrd.PatientPosition.F_FDR,
        "FFDL": mrd.PatientPosition.F_FDL,
    }

    # Check if the input string matches a key in the dictionary
    if s in position_map:
        return position_map[s]
    else:
        raise ValueError(f"Unknown Patient Position: {s}")


def convert_three_dimensional_float(three_dimensional_float, t=None):
    if t is None:
        t = (
            mrd.ThreeDimensionalFloat()
        )  # Replace with actual mrd.ThreeDimensionalFloat initialization
    t.x = three_dimensional_float.x
    t.y = three_dimensional_float.y
    t.z = three_dimensional_float.z
    return t


def convert_measurement_dependency(measurement_dependency, m=None):
    if m is None:
        m = mrd.MeasurementDependencyType()
    m.measurement_id = measurement_dependency.measurementID
    m.dependency_type = measurement_dependency.dependencyType
    return m


def convert_measurement_information(measurement_information, m=None):
    if m is None:
        m = mrd.MeasurementInformationType()

    if measurement_information.measurementID:
        m.measurement_id = measurement_information.measurementID

    if measurement_information.seriesDate:
        m.series_date = date_from_string(measurement_information.seriesDate)

    if measurement_information.seriesTime:
        m.series_time = time_from_string(measurement_information.seriesTime)

    m.patient_position = patient_position_from_string(
        measurement_information.patientPosition.value.upper()
    )

    if measurement_information.relativeTablePosition:
        m.relative_table_position = convert_measurement_information(
            measurement_information.relativeTablePosition
        )

    if measurement_information.initialSeriesNumber:
        m.initial_series_number = measurement_information.initialSeriesNumber

    if measurement_information.protocolName:
        m.protocol_name = measurement_information.protocolName

    if measurement_information.sequenceName:
        m.sequence_name = measurement_information.sequenceName

    if measurement_information.seriesDescription:
        m.series_description = measurement_information.seriesDescription

    # Loop through measurement dependencies and convert each
    for dependency in measurement_information.measurementDependency:
        m.measurement_dependency.append(convert_measurement_dependency(dependency))

    if measurement_information.seriesInstanceUIDRoot:
        m.series_instance_uid_root = measurement_information.seriesInstanceUIDRoot

    if measurement_information.frameOfReferenceUID:
        m.frame_of_reference_uid = measurement_information.frameOfReferenceUID

    # Handle referenced image sequence
    if measurement_information.referencedImageSequence:
        referenced_image = mrd.ReferencedImageSequenceType()
        for image in measurement_information.referencedImageSequence:
            referenced_image.referenced_sop_instance_uid.append(
                image.referencedSOPInstanceUID
            )
        m.referenced_image_sequence = referenced_image

    return m


def convert_acquisition_system_information(a, asi=None):
    if asi is None:
        asi = mrd.AcquisitionSystemInformationType()

    if a.systemVendor:
        asi.system_vendor = a.systemVendor

    if a.systemModel:
        asi.system_model = a.systemModel

    if a.systemFieldStrength_T:
        asi.system_field_strength_t = a.systemFieldStrength_T

    if a.relativeReceiverNoiseBandwidth:
        asi.relative_receiver_noise_bandwidth = a.relativeReceiverNoiseBandwidth

    if a.receiverChannels:
        asi.receiver_channels = a.receiverChannels

    if len(a.coilLabel) > 0:
        for c in a.coilLabel:
            cl = mrd.CoilLabelType()
            cl.coil_name = c.coilName
            cl.coil_number = c.coilNumber
            asi.coil_label.append(cl)

    if a.institutionName:
        asi.institution_name = a.institutionName

    if a.stationName:
        asi.station_name = a.stationName

    if a.deviceID:
        asi.device_id = a.deviceID

    if a.deviceSerialNumber:
        asi.device_serial_number = a.deviceSerialNumber

    return asi


def convert_experimental_conditions(e, ec=None):
    if ec is None:
        ec = mrd.ExperimentalConditionsType()
    ec.h1resonance_frequency_hz = e.H1resonanceFrequency_Hz
    return ec


def convert_matrix_size(m, matrix_size=None):
    if matrix_size is None:
        matrix_size = mrd.MatrixSizeType()
    matrix_size.x = m.x
    matrix_size.y = m.y
    matrix_size.z = m.z
    return matrix_size


def convert_field_of_view_mm(f, field_of_view=None):
    if field_of_view is None:
        field_of_view = mrd.FieldOfViewMm()
    field_of_view.x = f.x
    field_of_view.y = f.y
    field_of_view.z = f.z
    return field_of_view


def convert_encoding_space(e, encoding_space=None):
    if encoding_space is None:
        encoding_space = mrd.EncodingSpaceType()
    encoding_space.matrix_size = convert_matrix_size(e.matrixSize)
    encoding_space.field_of_view_mm = convert_field_of_view_mm(e.fieldOfView_mm)
    return encoding_space


def convert_limit(l, limit=None):
    if limit is None:
        limit = mrd.LimitType()
    limit.minimum = l.minimum
    limit.maximum = l.maximum
    limit.center = l.center
    return limit


def convert_encoding_limits(e, encoding_limits=None):
    if encoding_limits is None:
        encoding_limits = mrd.EncodingLimitsType()

    if e.kspace_encoding_step_0:
        encoding_limits.kspace_encoding_step_0 = e.kspace_encoding_step_0

    if e.kspace_encoding_step_1:
        encoding_limits.kspace_encoding_step_1 = e.kspace_encoding_step_1

    if e.kspace_encoding_step_2:
        encoding_limits.kspace_encoding_step_2 = e.kspace_encoding_step_2

    if e.average:
        encoding_limits.average = e.average

    if e.slice:
        encoding_limits.slice = e.slice

    if e.contrast:
        encoding_limits.contrast = e.contrast

    if e.phase:
        encoding_limits.phase = e.phase

    if e.repetition:
        encoding_limits.repetition = e.repetition

    if e.set:
        encoding_limits.set = e.set

    if e.segment:
        encoding_limits.segment = e.segment

    for i in range(8):  # assuming `e.user` has 8 elements
        if hasattr(e, f"user_{i}"):
            setattr(encoding_limits, f"user_{i}", getattr(e, f"user_{i}"))

    return encoding_limits


def convert_user_parameter_long(u):
    return mrd.UserParameterLongType(name=u.name, value=u.value)


def convert_user_parameter_double(u):
    return mrd.UserParameterDoubleType(name=u.name, value=u.value)


def convert_user_parameter_string(u):
    return mrd.UserParameterStringType(name=u.name, value=u.value)


def convert_trajectory_description(t):
    user_parameter_long = [convert_user_parameter_long(u) for u in t.userParameterLong]
    user_parameter_double = [
        convert_user_parameter_double(u) for u in t.userParameterDouble
    ]
    user_parameter_string = [
        convert_user_parameter_string(u) for u in t.userParameterString
    ]
    return mrd.TrajectoryDescriptionType(
        identifier=t.identifier,
        user_parameter_long=user_parameter_long,
        user_parameter_double=user_parameter_double,
        user_parameter_string=user_parameter_string,
        comment=t.comment,
    )


def convert_acceleration_factor(a):
    return mrd.AccelerationFactorType(
        kspace_encoding_step_1=a.kspace_encoding_step_1,
        kspace_encoding_step_2=a.kspace_encoding_step_2,
    )


def calibration_mode_from_string(m):
    calibration_mode_map = {
        "embedded": mrd.CalibrationMode.EMBEDDED,
        "interleaved": mrd.CalibrationMode.INTERLEAVED,
        "separate": mrd.CalibrationMode.SEPARATE,
        "external": mrd.CalibrationMode.EXTERNAL,
        "other": mrd.CalibrationMode.OTHER,
    }

    # Check if the input string matches a key in the dictionary
    if m in calibration_mode_map:
        return calibration_mode_map[m]
    else:
        raise ValueError(f"Unknown CalibrationMode: {m}")


def interleaving_dimension_from_string(s):
    interleaving_dimension_map = {
        "phase": mrd.InterleavingDimension.PHASE,
        "repetition": mrd.InterleavingDimension.REPETITION,
        "contrast": mrd.InterleavingDimension.CONTRAST,
        "average": mrd.InterleavingDimension.AVERAGE,
        "other": mrd.InterleavingDimension.OTHER,
    }
    if s in interleaving_dimension_map:
        return interleaving_dimension_map[s]
    else:
        raise ValueError(f"Unknown InterleavingDimension: {s}")


def convert_multiband_spacing(m, multiband_spacing=None):
    if multiband_spacing is None:
        multiband_spacing = mrd.MultibandSpacingType()
    for s in m.dZ:
        multiband_spacing.d_z.append(s)
    return multiband_spacing


def convert_multiband_calibration_type(m):
    if m.value.upper() == "SEPARABLE2D":
        return mrd.Calibration.SEPARABLE_2D
    elif m.value.upper() == "FULL3D":
        return mrd.Calibration.FULL_3D
    elif m.value.upper() == "OTHER":
        return mrd.Calibration.OTHER
    else:
        raise ValueError("Unknown Calibration")


def convert_multiband(m, multiband=None):
    if multiband is None:
        multiband = mrd.MultibandType()
    for s in m.spacing:
        multiband.spacing.append(convert_multiband_spacing(s))
    multiband.delta_kz = m.deltaKz
    multiband.multiband_factor = m.multiband_factor
    multiband.calibration = convert_multiband_calibration_type(m.calibration)
    multiband.calibration_encoding = m.calibration_encoding
    return multiband


def convert_parallel_imaging(p, parallel_imaging=None):
    if parallel_imaging is None:
        parallel_imaging = mrd.ParallelImagingType()
    if p.accelerationFactor:
        parallel_imaging.acceleration_factor = convert_acceleration_factor(
            p.accelerationFactor
        )
    if p.calibrationMode:
        parallel_imaging.calibration_mode = calibration_mode_from_string(
            p.calibrationMode.value.lower()
        )
    if p.interleavingDimension:
        parallel_imaging.interleaving_dimension = interleaving_dimension_from_string(
            p.interleavingDimension.value.lower()
        )
    if p.multiband:
        parallel_imaging.multiband = convert_multiband(p.multiband)
    return parallel_imaging


def convert_encoding(e, encoding=None):
    if encoding is None:
        encoding = mrd.EncodingType()
    encoding.encoded_space = convert_encoding_space(e.encodedSpace)
    encoding.recon_space = convert_encoding_space(e.reconSpace)
    encoding.encoding_limits = convert_encoding_limits(e.encodingLimits)

    if e.trajectory.value.upper() == "CARTESIAN":
        encoding.trajectory == mrd.Trajectory.CARTESIAN
    elif e.trajectory.value.upper() == "EPI":
        encoding.trajectory == mrd.Trajectory.EPI
    elif e.trajectory.value.upper() == "RADIAL":
        encoding.trajectory == mrd.Trajectory.RADIAL
    elif e.trajectory.value.upper() == "GOLDENANGLE":
        encoding.trajectory == mrd.Trajectory.GOLDENANGLE
    elif e.trajectory.value.upper() == "SPIRAL":
        encoding.trajectory == mrd.Trajectory.SPIRAL
    elif e.trajectory.value.upper() == "OTHER":
        encoding.trajectory == mrd.Trajectory.OTHER
    else:
        raise RuntimeError("Unknown TrajectoryType")

    if e.trajectoryDescription:
        encoding.trajectory_description = convert_trajectory_description(
            e.trajectoryDescription
        )
    if e.parallelImaging:
        encoding.parallel_imaging = convert_parallel_imaging(e.parallelImaging)
    if e.echoTrainLength:
        encoding.echo_train_length = e.echoTrainLength

    return encoding


def convert_diffusion_dimension(diffusion_dimension):
    if diffusion_dimension.value.upper() == "AVERAGE":
        return mrd.DiffusionDimension.AVERAGE
    elif diffusion_dimension.value.upper() == "CONTRAST":
        return mrd.DiffusionDimension.CONTRAST
    elif diffusion_dimension.value.upper() == "PHASE":
        return mrd.DiffusionDimension.PHASE
    elif diffusion_dimension.value.upper() == "REPETITION":
        return mrd.DiffusionDimension.REPETITION
    elif diffusion_dimension.value.upper() == "SET":
        return mrd.DiffusionDimension.SET
    elif diffusion_dimension.value.upper() == "SEGMENT":
        return mrd.DiffusionDimension.SEGMENT
    elif diffusion_dimension.value.upper() == "USER_0":
        return mrd.DiffusionDimension.USER0
    elif diffusion_dimension.value.upper() == "USER_1":
        return mrd.DiffusionDimension.USER1
    elif diffusion_dimension.value.upper() == "USER_2":
        return mrd.DiffusionDimension.USER2
    elif diffusion_dimension.value.upper() == "USER_3":
        return mrd.DiffusionDimension.USER3
    elif diffusion_dimension.value.upper() == "USER_4":
        return mrd.DiffusionDimension.USER4
    elif diffusion_dimension.value.upper() == "USER_5":
        return mrd.DiffusionDimension.USER5
    elif diffusion_dimension.value.upper() == "USER_6":
        return mrd.DiffusionDimension.USER6
    elif diffusion_dimension.value.upper() == "USER_7":
        return mrd.DiffusionDimension.USER7
    else:
        raise ValueError("Unknown diffusion dimension")


def convert_gradient_direction(g, gradient_direction=None):
    if gradient_direction is None:
        gradient_direction = mrd.GradientDirectionType()
    gradient_direction.rl = g.rl
    gradient_direction.ap = g.ap
    gradient_direction.fh = g.fh
    return gradient_direction


def convert_diffusion(d, diffusion=None):
    if diffusion is None:
        diffusion = mrd.DiffusionType()
    diffusion.gradient_direction = convert_gradient_direction(d.gradient_direction)
    diffusion.bvalue = d.bvalue
    return diffusion


def convert_sequence_parameters(s, sequence_parameters=None):
    if sequence_parameters is None:
        sequence_parameters = mrd.SequenceParametersType()

    if s.TR:
        sequence_parameters.t_r.extend(s.TR)

    if s.TE:
        sequence_parameters.t_e.extend(s.TE)

    if s.TI:
        sequence_parameters.t_i.extend(s.TI)

    if s.flipAngle_deg:
        sequence_parameters.flip_angle_deg.extend(s.flipAngle_deg)

    if s.sequence_type:
        sequence_parameters.sequence_type = s.sequence_type

    if s.echo_spacing:
        sequence_parameters.echo_spacing.extend(s.echo_spacing)

    if s.diffusionDimension:
        sequence_parameters.diffusion_dimension = convert_diffusion_dimension(
            s.diffusionDimension
        )

    if s.diffusion:
        sequence_parameters.diffusion = [convert_diffusion(d) for d in s.diffusion]

    if s.diffusionScheme:
        sequence_parameters.diffusion_scheme = s.diffusionScheme

    return sequence_parameters


def convert_userbase64(u):
    return mrd.UserParameterBase64Type(name=u.name, value=u.value)


def convert_user_parameters(u, user_parameters=None):
    if user_parameters is None:
        user_parameters = mrd.UserParametersType()

    # Assuming userParameterLong, userParameterDouble, userParameterString, userParameterBase64 are lists
    for p in u.userParameterLong:
        user_parameters.user_parameter_long.append(convert_user_parameter_long(p))

    for p in u.userParameterDouble:
        user_parameters.user_parameter_double.append(convert_user_parameter_double(p))

    for p in u.userParameterString:
        user_parameters.user_parameter_string.append(convert_user_parameter_string(p))

    for p in u.userParameterBase64:
        user_parameters.user_parameter_base64.append(convert_userbase64(p))

    return user_parameters


def convert_waveform_type(w):
    if w.value.upper() == "ECG":
        return mrd.WaveformType.ECG
    elif w.value.upper() == "PULSE":
        return mrd.WaveformType.PULSE
    elif w.value.upper() == "RESPIRATORY":
        return mrd.WaveformType.RESPIRATORY
    elif w.value.upper() == "TRIGGER":
        return mrd.WaveformType.TRIGGER
    elif w.value.upper() == "GRADIENTWAVEFORM":
        return mrd.WaveformType.GRADIENTWAVEFORM
    elif w.value.upper() == "OTHER":
        return mrd.WaveformType.OTHER
    else:
        raise ValueError(f"Unknown waveform type: {w}")


def convert_waveform_information(w, waveform_information=None):
    if waveform_information is None:
        waveform_information = mrd.WaveformInformationType()
    waveform_information.waveform_name = w.waveformName
    waveform_information.waveform_type = convert_waveform_type(w.waveformType)

    if w.userParameters:
        waveform_information.user_parameters = convert_user_parameters(w.userParameters)

    return waveform_information


def convert_encoding_counters(e, encoding_counters=None):
    if encoding_counters is None:
        encoding_counters = mrd.EncodingCounters()
    encoding_counters.kspace_encode_step_1 = e.kspace_encode_step_1
    encoding_counters.kspace_encode_step_2 = e.kspace_encode_step_2
    encoding_counters.average = e.average
    encoding_counters.slice = e.slice
    encoding_counters.contrast = e.contrast
    encoding_counters.phase = e.phase
    encoding_counters.repetition = e.repetition
    encoding_counters.set = e.set
    encoding_counters.segment = e.segment

    if hasattr(e, "user") and e.user:
        encoding_counters.user.extend(e.user)

    return encoding_counters
