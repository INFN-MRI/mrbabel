"""GEHC to MRD Conversion Utilities."""

__all__ = [
    "read_gehc_header",
    "read_gehc_acquisitions",
]

import base64
import copy

from types import SimpleNamespace

import numpy as np

import mrd
import getools

from ...utils import get_user_param

from ._dicom2mrd import read_dicom_header


def read_gehc_header(
    gehc_head: dict,
    head_template: mrd.Header | None = None,
    acquisitions_template: list[mrd.Acquisition] | None = None,
) -> mrd.Header:
    """Create MRD Header from a GEHC file."""
    if acquisitions_template is not None:
        dset = getools.raw2dicom(
            gehc_head, False, head_template, acquisitions_template[0]
        )
    else:
        dset = getools.raw2dicom(gehc_head, False, head_template, None)
    head = read_dicom_header(dset)

    # missing info
    head.acquisition_system_information.receiver_channels = int(
        gehc_head["rdb_hdr"]["dab"][1] - gehc_head["rdb_hdr"]["dab"][0] + 1
    )
    head.acquisition_system_information.device_id = gehc_head["exam"][
        "service_id"
    ].strip()
    head.acquisition_system_information.device_serial_number = gehc_head["exam"][
        "uniq_sys_id"
    ]
    head.acquisition_system_information.relative_receiver_noise_bandwidth = 1.0
    for n in range(head.acquisition_system_information.receiver_channels):
        coil_label = mrd.CoilLabelType(
            coil_number=n, coil_name=dset.ReceiveCoilName + f"_{n}"
        )
        head.acquisition_system_information.coil_label.append(coil_label)
    head.measurement_information.initial_series_number = int(
        head.measurement_information.initial_series_number
    )
    head.measurement_information.sequence_name = gehc_head["image"]["psd_iname"].strip()
    head.measurement_information.relative_table_position = mrd.ThreeDimensionalFloat()
    head.measurement_information.relative_table_position.x = 0.0
    head.measurement_information.relative_table_position.y = 0.0
    head.measurement_information.relative_table_position.z = gehc_head["series"][
        "tablePosition"
    ]
    head.measurement_information.series_instance_uid_root = dset.SeriesInstanceUID[:20]
    head.user_parameters.user_parameter_base64.append(
        mrd.UserParameterBase64Type(
            name="DicomJson",
            value=base64.b64encode(dset.to_json().encode("utf-8")).decode("utf-8"),
        )
    )

    # update with blueprint
    if head_template is not None:
        psdname = gehc_head["image"]["psd_iname"].strip()
        is_fidall = (
            "fidall" in psdname
            or "3drad" in psdname
            or "silen" in psdname
            or "burzte" in psdname
        )

        # replace encoding
        encoding = head_template.encoding
        if is_fidall:
            # update encoded space in-plane
            encoding[-1].encoded_space.field_of_view_mm.x = gehc_head["image"]["dfov"]
            encoding[-1].encoded_space.field_of_view_mm.y = gehc_head["image"]["dfov"]
            encoding[-1].recon_space.field_of_view_mm.x = gehc_head["image"]["dfov"]
            encoding[-1].recon_space.field_of_view_mm.y = gehc_head["image"]["dfov"]

            imode = get_user_param(head_template, "ImagingMode")
            if imode is None:
                imode = dset.MRAcquisitionType
            if imode == "3Dnoncart":
                encoding[-1].encoded_space.field_of_view_mm.z = gehc_head["image"][
                    "dfov"
                ]
                encoding[-1].recon_space.field_of_view_mm.z = gehc_head["image"]["dfov"]
            elif "hybrid" in imode:
                encoding[-1].encoded_space.field_of_view_mm.z = get_user_param(
                    head, "SliceThickness"
                )
                encoding[-1].recon_space.field_of_view_mm.z = get_user_param(
                    head, "SliceThickness"
                )
            else:
                nz = gehc_head["image"]["slquant"]
                spacing = get_user_param(head, "SpacingBetweenSlices")
                encoding[-1].encoded_space.field_of_view_mm.z = spacing * nz
                encoding[-1].recon_space.field_of_view_mm.z = spacing * nz

                # update matrix size
                encoding[-1].encoded_space.matrix_size.z = nz
                encoding[-1].recon_space.matrix_size.z = nz
        head.encoding = encoding

        # replace contrast
        head.sequence_parameters = head_template.sequence_parameters

        # update user parameters
        if is_fidall:
            head.user_parameters.user_parameter_base64.extend(
                head_template.user_parameters.user_parameter_base64
            )
            head.user_parameters.user_parameter_double.extend(
                head_template.user_parameters.user_parameter_double
            )
            head.user_parameters.user_parameter_long.extend(
                head_template.user_parameters.user_parameter_long
            )
            head.user_parameters.user_parameter_string.extend(
                head_template.user_parameters.user_parameter_string
            )
        else:
            head.user_parameters = head_template.user_parameters

    # insert number of dimensions
    if get_user_param(head, "ImagingMode") is None:
        head.user_parameters.user_parameter_string.append(
            mrd.UserParameterStringType(
                name="ImagingMode", value=dset.MRAcquisitionType
            )
        )

    return head


def read_gehc_acquisitions(
    gehc_head, gehc_data, head_template=None, acquisitions_template=None
):
    """Create a list of MRD Acquisitions from a list of Siemens Acquisitions."""
    dset = getools.raw2dicom(gehc_head, False, head_template)
    n_dims = get_dimension(dset, acquisitions_template)
    n_views = int(gehc_head["rdb_hdr"]["da_yres"])
    n_echoes = int(gehc_head["rdb_hdr"]["nechoes"])
    n_slices = int(gehc_head["rdb_hdr"]["nslices"])
    n_acquisitions = len(gehc_data)
    center_sample = int(gehc_head["rdb_hdr"]["frame_size"] // 2)
    vecs = [
        get_slice_vectors(gehc_head, n) for n in range(len(gehc_head["data_acq_tab"]))
    ]
    commons = SimpleNamespace(
        n_dims=n_dims,
        n_views=n_views,
        n_echoes=n_echoes,
        n_slices=n_slices,
        n_frames=n_acquisitions,
        center_sample=center_sample,
        vecs=vecs,
    )
    acquisitions = [
        read_gehc_acquistion(gehc_data[n], gehc_head, commons)
        for n in range(n_acquisitions)
    ]

    # update
    if acquisitions_template is not None:
        psdname = gehc_head["image"]["psd_iname"].strip()
        is_fidall = (
            "fidall" in psdname
            or "3drad" in psdname
            or "silen" in psdname
            or "burzte" in psdname
        )

        # Split echoes along readout
        if get_user_param(head_template, "ReadoutLength"):
            data = np.stack([acq.data for acq in acquisitions], axis=0)

            # get actual number of pointes and contrasts
            n_pts = get_user_param(head_template, "ReadoutLength")
            n_contrasts = len(np.unique(head_template.sequence_parameters.t_e))

            for n in range(n_acquisitions):
                acquisitions[n].data = None
            acquisitions = np.stack(
                [copy.deepcopy(acquisitions) for n in range(n_contrasts)]
            ).ravel()
            n_acquisitions = len(acquisitions)

            data = data[..., :n_pts]
            data = data.reshape(*data.shape[:-1], n_contrasts, -1)
            data = data.swapaxes(1, 2)
            data = data.reshape(-1, *data.shape[-2:])
            data = np.ascontiguousarray(data)

            for n in range(n_acquisitions):
                acquisitions[n].data = data[n]

        for n in range(n_acquisitions):
            acquisitions[n].trajectory = acquisitions_template[n].trajectory
            acquisitions[n].head.flags = acquisitions_template[n].head.flags
            acquisitions[n].head.idx.kspace_encode_step_1 = acquisitions_template[
                n
            ].head.idx.kspace_encode_step_1
            acquisitions[n].head.idx.kspace_encode_step_2 = acquisitions_template[
                n
            ].head.idx.kspace_encode_step_2
            acquisitions[n].head.idx.slice = acquisitions_template[n].head.idx.slice
            acquisitions[n].head.idx.contrast = acquisitions_template[
                n
            ].head.idx.contrast
            acquisitions[n].head.discard_pre = acquisitions_template[n].head.discard_pre
            if is_fidall:
                acquisitions[n].head.discard_post = (
                    acquisitions[n].samples()
                    - acquisitions_template[n].head.discard_post
                    - 1
                )
            else:
                acquisitions[n].head.discard_post = acquisitions_template[
                    n
                ].head.discard_post
            acquisitions[n].head.center_sample = acquisitions_template[
                n
            ].head.center_sample
            acquisitions[n].head.encoding_space_ref = acquisitions_template[
                n
            ].head.encoding_space_ref
            acquisitions[n].head.sample_time_us = acquisitions_template[
                n
            ].head.sample_time_us
            acquisitions[n].head.scan_counter = acquisitions_template[
                n
            ].head.scan_counter

    return acquisitions


def read_gehc_acquistion(gehc_acquisition, gehc_head, common) -> mrd.Acquisition:
    """Create MRD Acquisition from a GEHC Acquisition."""
    acquisition = mrd.Acquisition()

    # Fill in the header fields
    # Flags
    defs = mrd.AcquisitionFlags
    if gehc_acquisition.viewNum == 0:
        acquisition.head.flags = defs.FIRST_IN_ENCODE_STEP_1
    if gehc_acquisition.viewNum == common.n_views - 1:
        acquisition.head.flags = defs.LAST_IN_ENCODE_STEP_1
    if common.n_dims == 2:
        if gehc_acquisition.sliceNum == 0:
            acquisition.head.flags = defs.FIRST_IN_SLICE
        if gehc_acquisition.sliceNum == common.n_slices - 1:
            acquisition.head.flags = defs.LAST_IN_SLICE
    if common.n_dims == 3:
        if gehc_acquisition.sliceNum == 0:
            acquisition.head.flags = defs.FIRST_IN_ENCODE_STEP_2
        if gehc_acquisition.sliceNum == common.n_slices - 1:
            acquisition.head.flags = defs.LAST_IN_ENCODE_STEP_2
    if gehc_acquisition.echoNum == 0:
        acquisition.head.flags = defs.FIRST_IN_CONTRAST
    if gehc_acquisition.echoNum == common.n_echoes - 1:
        acquisition.head.flags = defs.LAST_IN_CONTRAST
    if gehc_acquisition.FrameCount == common.n_frames - 1:
        acquisition.head.flags = defs.LAST_IN_MEASUREMENT

    # Encoding Counter
    encoding_counter = mrd.EncodingCounters()
    encoding_counter.kspace_encode_step_1 = gehc_acquisition.viewNum
    if common.n_dims == 3:
        encoding_counter.kspace_encode_step_2 = gehc_acquisition.sliceNum
        encoding_counter.slice = 0
    else:
        encoding_counter.kspace_encode_step_2 = 0
        encoding_counter.slice = gehc_acquisition.sliceNum
    encoding_counter.contrast = gehc_acquisition.echoNum
    encoding_counter.user.extend(
        [
            gehc_acquisition.opcode,
            gehc_acquisition.operation,
            0,
            0,
            0,
        ]
    )

    acquisition.head.idx = encoding_counter
    acquisition.head.scan_counter = gehc_acquisition.FrameCount
    for n in range(gehc_acquisition.Data.shape[0]):
        acquisition.head.channel_order.append(n)

    # Readout
    acquisition.head.discard_pre = 0
    acquisition.head.discard_post = 0
    acquisition.head.center_sample = common.center_sample
    acquisition.head.encoding_space_ref = 0
    acquisition.head.sample_time_us = 2.0

    # Geometry
    acquisition.head.position = common.vecs[gehc_acquisition.sliceNum].center
    acquisition.head.read_dir = common.vecs[gehc_acquisition.sliceNum].read_dir
    acquisition.head.phase_dir = common.vecs[gehc_acquisition.sliceNum].phase_dir
    acquisition.head.slice_dir = common.vecs[gehc_acquisition.sliceNum].slice_dir
    patient_table_position = [
        0,
        0,
        gehc_head["series"]["table_entry"],
    ]
    acquisition.head.patient_table_position = patient_table_position

    # Data
    acquisition.data = gehc_acquisition.Data

    return acquisition


# %% utils
def get_slice_vectors(gehc_head, slice_number):
    slice_table = gehc_head["data_acq_tab"][slice_number]

    patient_entry = gehc_head["series"]["entry"] - 1
    patient_position = gehc_head["series"]["position"] - 1
    if patient_position:
        patient_position = int(np.log2(patient_position))

    # get corners
    gwp1 = rotate_vector_on_patient(
        patient_entry, patient_position, slice_table["gw_point1"]
    )
    gwp2 = rotate_vector_on_patient(
        patient_entry, patient_position, slice_table["gw_point2"]
    )
    gwp3 = rotate_vector_on_patient(
        patient_entry, patient_position, slice_table["gw_point3"]
    )

    # get directions
    read_dir, phase_dir, slice_dir = make_direction_vectors(gwp1, gwp2, gwp3)
    center = 3 * [0.0]
    center[0] = (gwp3[0] + gwp2[0]) / 2.0
    center[1] = (gwp3[1] + gwp2[1]) / 2.0
    center[2] = (gwp3[2] + gwp2[2]) / 2.0

    # create output
    vecs = SimpleNamespace(
        read_dir=read_dir,
        phase_dir=phase_dir,
        slice_dir=slice_dir,
        center=center,
    )

    return vecs


def rotate_vector_on_patient(entry, pos, input):
    """
    Rotate based on patient position and convert to patient coordinate system by swapping signs of X and Y.

    Params
    ------
    entry : int
        0="Head First", 1="Feet First"
    pos : int
        0="Supine", 1="Prone", 2="Decubitus Left", 3="Decubitus Right"
    input : np.ndarray
        original direction vector

    """
    rot_hfs = np.asarray([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    rot_hfp = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    rot_hfdl = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    rot_hfdr = np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    rot_ffs = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    rot_ffp = np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    rot_ffdl = np.asarray([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float32)
    rot_ffdr = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float32)

    patient_rotations = [
        [rot_hfs, rot_hfp, rot_hfdl, rot_hfdr],
        [rot_ffs, rot_ffp, rot_ffdl, rot_ffdr],
    ]

    rotation_matrix = patient_rotations[entry][pos]
    return rotation_matrix @ input


def make_direction_vectors(gwp1, gwp2, gwp3):
    """
     Calculate read, phase, and slice direction vectors from three corners of plane.

    | r1 p1 s1 |   | x1 y1 z1 |
    | r2 p2 s2 | = | x2 y2 z2 |
    | r3 p3 s3 |   | x3 y3 z3 |

    """
    x1, y1, z1 = gwp1
    x2, y2, z2 = gwp2
    x3, y3, z3 = gwp3

    r1, r2, r3 = x2 - x1, y2 - y1, z2 - z1
    xd = (r1**2 + r2**2 + r3**2) ** 0.5

    p1, p2, p3 = x3 - x1, y3 - y1, z3 - z1
    yd = (p1**2 + p2**2 + p3**2) ** 0.5

    s1, s2, s3 = r2 * p3 - r3 * p2, r3 * p1 - r1 * p3, r1 * p2 - r2 * p1
    zd = (s1**2 + s2**2 + s3**2) ** 0.5

    if xd == 0.0:
        r1, r2, r3, xd = 1.0, 0.0, 0.0, 1.0
    if yd == 0.0:
        p1, p2, p3, yd = 0.0, 1.0, 0.0, 1.0
    if zd == 0.0:
        s1, s2, s3, zd = 0.0, 0.0, 1.0, 1.0

    read_dir = [r1 / xd, r2 / xd, r3 / xd]
    phase_dir = [p1 / yd, p2 / yd, p3 / yd]
    slice_dir = [s1 / zd, s2 / zd, s3 / zd]

    return read_dir, phase_dir, slice_dir


def get_dimension(dset, acquisition):
    """Get number of k-space dimensions."""
    if acquisition is not None:
        if acquisition[0].trajectory_dimensions():
            return acquisition[0].trajectory_dimensions()
    return int(dset.MRAcquisitionType[0])
