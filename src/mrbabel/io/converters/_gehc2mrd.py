"""GEHC to MRD Conversion Utilities."""

__all__ = [
    "read_gehc_header",
    # "read_gehc_acquisitions",
]

import mrd
import getools

from ._dicom2mrd import read_dicom_header


def read_gehc_header(
    gehc_hdr: dict, mrd_template: mrd.Header | None = None
) -> mrd.Header:
    """Create MRD Header from a GEHC file."""
    dset = getools.raw2dicom(gehc_hdr, False, mrd_template)
    mrd_hdr = read_dicom_header(dset)


def read_gehc_acquisitions(): ...


def read_gehc_acquistion(twix_acquisition, twix_hdr, enc_ref) -> mrd.Acquisition:
    """Create MRD Acquisition from a Siemens Acquisition."""
    acquisition = mrd.Acquisition()

    # Fill in the header fields
    defs = mrd.AcquisitionFlags
    if twix_acquisition.mdh.EvalInfoMask & (1 << 25):
        acquisition.flags = defs.IS_NOISE_MEASUREMENT
    if twix_acquisition.mdh.EvalInfoMask & (1 << 28):
        acquisition.flags = defs.FIRST_IN_SLICE
    if twix_acquisition.mdh.EvalInfoMask & (1 << 29):
        acquisition.flags = defs.LAST_IN_SLICE
    if twix_acquisition.mdh.EvalInfoMask & (1 << 11):
        acquisition.flags = defs.LAST_IN_REPETITION

    # if a line is both image and ref, then do not set the ref flag
    if twix_acquisition.mdh.EvalInfoMask & (1 << 23):
        acquisition.flags = defs.IS_PARALLEL_CALIBRATION_AND_IMAGING
    else:
        if twix_acquisition.mdh.EvalInfoMask & (1 << 22):
            acquisition.flags = defs.IS_PARALLEL_CALIBRATION

    if twix_acquisition.mdh.EvalInfoMask & (1 << 24):
        acquisition.flags = defs.IS_REVERSE
    if twix_acquisition.mdh.EvalInfoMask & (1 << 11):
        acquisition.flags = defs.LAST_IN_MEASUREMENT
    if twix_acquisition.mdh.EvalInfoMask & (1 << 21):
        acquisition.flags = defs.IS_PHASECORR_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 1):
        acquisition.flags = defs.IS_NAVIGATION_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 1):
        acquisition.flags = defs.IS_RTFEEDBACK_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 2):
        acquisition.flags = defs.IS_HPFEEDBACK_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 51):
        acquisition.flags = defs.IS_DUMMYSCAN_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 10):
        acquisition.flags = defs.IS_SURFACECOILCORRECTIONSCAN_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 5):
        acquisition.flags = defs.IS_DUMMYSCAN_DATA

    if twix_acquisition.mdh.EvalInfoMask & (1 << 46):
        acquisition.flags = defs.LAST_IN_MEASUREMENT

    encoding_counter = mrd.EncodingCounters()
    encoding_counter.kspace_encode_step_1 = twix_acquisition.mdh.Counter.Lin
    encoding_counter.kspace_encode_step_2 = twix_acquisition.mdh.Counter.Par
    encoding_counter.average = twix_acquisition.mdh.Counter.Ave
    encoding_counter.slice = twix_acquisition.mdh.Counter.Sli
    encoding_counter.contrast = twix_acquisition.mdh.Counter.Eco
    encoding_counter.phase = twix_acquisition.mdh.Counter.Phs
    encoding_counter.repetition = twix_acquisition.mdh.Counter.Rep
    encoding_counter.set = twix_acquisition.mdh.Counter.Set
    encoding_counter.segment = twix_acquisition.mdh.Counter.Seg
    encoding_counter.user.extend(
        [
            twix_acquisition.mdh.Counter.Ida,
            twix_acquisition.mdh.Counter.Idb,
            twix_acquisition.mdh.Counter.Idc,
            twix_acquisition.mdh.Counter.Idd,
            twix_acquisition.mdh.Counter.Ide,
        ]
    )

    acquisition.head.idx = encoding_counter
    acquisition.head.measurement_uid = twix_acquisition.mdh.MeasUID
    acquisition.head.scan_counter = twix_acquisition.mdh.ScanCounter
    acquisition.head.acquisition_time_stamp = twix_acquisition.mdh.TimeStamp
    acquisition.head.physiology_time_stamp = twix_acquisition.mdh.PMUTimeStamp
    for n in range(twix_acquisition.mdh.UsedChannels):
        acquisition.head.channel_order.append(n)

    acquisition.head.discard_pre = int(twix_acquisition.mdh.CutOff.Pre)
    acquisition.head.discard_post = int(twix_acquisition.mdh.CutOff.Post)
    acquisition.head.center_sample = int(twix_acquisition.mdh.CenterCol)
    acquisition.head.encoding_space_ref = enc_ref
    acquisition.head.sample_time_us = (
        twix_hdr["MeasYaps"]["sRXSPEC"]["alDwellTime"][0] / 1000.0
    )

    position = [
        twix_acquisition.mdh.SliceData.SlicePos.Sag,
        twix_acquisition.mdh.SliceData.SlicePos.Cor,
        twix_acquisition.mdh.SliceData.SlicePos.Tra,
    ]
    acquisition.head.position = position

    quaternion = [
        twix_acquisition.mdh.SliceData.Quaternion[1],
        twix_acquisition.mdh.SliceData.Quaternion[2],
        twix_acquisition.mdh.SliceData.Quaternion[3],
        twix_acquisition.mdh.SliceData.Quaternion[0],
    ]
    read_dir, phase_dir, slice_dir = quat.quaternion_to_directions(quaternion)
    acquisition.head.read_dir = read_dir
    acquisition.head.phase_dir = phase_dir
    acquisition.head.slice_dir = slice_dir

    patient_table_position = [
        twix_acquisition.mdh.PTABPosX,
        twix_acquisition.mdh.PTABPosY,
        twix_acquisition.mdh.PTABPosZ,
    ]
    acquisition.head.patient_table_position = patient_table_position

    acquisition.head.user_int.extend(list(twix_acquisition.mdh.IceProgramPara[:7]))
    acquisition.head.user_int.append(twix_acquisition.mdh.TimeSinceLastRF)
    acquisition.head.user_float.extend(list(twix_acquisition.mdh.IceProgramPara[8:16]))

    # # Resize the data structure (for example, using numpy arrays or lists)
    acquisition.data = twix_acquisition.data

    # # If trajectory dimensions are present, resize and fill the trajectory data
    # if acq.trajectory_dimensions > 0:
    #     acquisition.trajectory = acq.traj

    return acquisition
