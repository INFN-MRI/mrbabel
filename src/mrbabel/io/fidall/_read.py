"""Fidall metadata reading routines."""

__all__ = ["read_fidall"]

import base64

import numpy as np
import mrd

try:
    import getools

    __GEHC_AVAILABLE__ = True
except Exception:
    __GEHC_AVAILABLE__ = False

from ...utils._serialization import serialize_array


def read_fidall(
    filepath: str,
    dcfpath: str | None = None,
    methodpath: str | None = None,
    sliceprofpath: str | None = None,
) -> tuple[mrd.Header, list[mrd.Acquisition]]:
    """
    Read input Fidall method file.

    Parameters
    ----------
    filepath : str
        Path to the file.
    dcfpath : str, optional
        Path to the dcf file, if not in ``filepath``.
        The default is ``None`` - will search in ``"filepath_dcf.mat"``
    methodpath : str, optional
        Path to the schedule description file, if not in ``filepath``.
        The default is ``None`` - will search in ``"filepath_method.mat"``
    sliceprofpath : str, optional
        Path to the slice profile file, if not in ``filepath``.
        The default is `None`` - will search in ``"filepath_sliceprof.mat"`

    Returns
    -------
    meta : list[mrd.Acquisition]
        Data-less MRD Acquisitions parsed from Fidall method file.
    head : mrd.Head
        MRD Header parsed from Fidall method file.

    """
    if __GEHC_AVAILABLE__ is False:
        raise ValueError(
            "GEHC reader is private - ask for access at https://docs.google.com/forms/d/1BvA1h8qb9GmndqiXMplQbf3IujgBIehQ1psnfmW0tew/edit"
        )
    meta = getools.fidall.read_method(filepath, dcfpath, methodpath, sliceprofpath)

    # initialize mrd Header
    head = mrd.Header(version=2.0)

    # fill spatial info
    enc = mrd.EncodingType()
    enc.trajectory = mrd.Trajectory.OTHER

    # encoded space
    enc.encoded_space.matrix_size.x = meta.shape[-1]
    enc.encoded_space.matrix_size.y = meta.shape[-2]
    if len(meta.shape) == 3:
        enc.encoded_space.matrix_size.z = meta.shape[-3]
    else:
        enc.encoded_space.matrix_size.z = 1
    enc.encoded_space.field_of_view_mm.x = meta.shape[-1] * meta.resolution[-1]
    enc.encoded_space.field_of_view_mm.y = meta.shape[-2] * meta.resolution[-2]
    if len(meta.shape) == 3:
        enc.encoded_space.field_of_view_mm.z = meta.shape[-3] * meta.resolution[-3]
    else:
        enc.encoded_space.field_of_view_mm.z = 1.0

    # recon space
    enc.recon_space.matrix_size.x = meta.shape[-1]
    enc.recon_space.matrix_size.y = meta.shape[-2]
    if len(meta.shape) == 3:
        enc.recon_space.matrix_size.z = meta.shape[-3]
    else:
        enc.recon_space.matrix_size.z = 1
    enc.recon_space.field_of_view_mm.x = meta.shape[-1] * meta.resolution[-1]
    enc.recon_space.field_of_view_mm.y = meta.shape[-2] * meta.resolution[-2]
    if len(meta.shape) == 3:
        enc.recon_space.field_of_view_mm.z = meta.shape[-3] * meta.resolution[-3]
    else:
        enc.recon_space.field_of_view_mm.z = 1.0

    # encoding  limits
    enc.encoding_limits.kspace_encoding_step_0 = mrd.LimitType()
    enc.encoding_limits.kspace_encoding_step_0.maximum = int(meta.traj.shape[-2]) - 1
    enc.encoding_limits.kspace_encoding_step_0.center = int(meta.traj.shape[-2]) // 2

    enc.encoding_limits.kspace_encoding_step_1 = mrd.LimitType()
    enc.encoding_limits.kspace_encoding_step_1.maximum = int(meta.traj.shape[-3]) - 1
    enc.encoding_limits.kspace_encoding_step_1.center = int(meta.traj.shape[-3]) // 2

    if meta.user["ImagingMode"] != "3Dnoncart":
        if meta.traj.shape[-1] == 2:
            enc.encoding_limits.slice = mrd.LimitType()
            enc.encoding_limits.slice.maximum = int(enc.encoded_space.matrix_size.z) - 1
            enc.encoding_limits.slice.center = int(enc.encoded_space.matrix_size.z) // 2
        else:
            enc.encoding_limits.kspace_encoding_step_2 = mrd.LimitType()
            enc.encoding_limits.kspace_encoding_step_2.maximum = (
                int(enc.encoded_space.matrix_size.z) - 1
            )
            enc.encoding_limits.kspace_encoding_step_2.center = (
                int(enc.encoded_space.matrix_size.z) // 2
            )

    enc.encoding_limits.contrast = mrd.LimitType()
    enc.encoding_limits.contrast.maximum = int(meta.traj.shape[-4]) - 1
    enc.encoding_limits.contrast.center = int(meta.traj.shape[-4]) // 2

    # append
    head.encoding.append(enc)

    # sequence parameters
    seq = mrd.SequenceParametersType()
    seq.flip_angle_deg = np.atleast_1d(meta.method.VariableFlip).tolist()
    seq.t_e = np.atleast_1d(meta.method.VariableTE).tolist()
    seq.t_r = np.atleast_1d(meta.method.VariableTR).tolist()
    seq.t_i = np.atleast_1d(meta.method.InversionTime).tolist()
    head.sequence_parameters = seq

    # user parameters
    head.user_parameters = mrd.UserParametersType()

    for key, value in meta.user.items():
        if isinstance(value, str):
            head.user_parameters.user_parameter_string.append(
                mrd.UserParameterStringType(name=key, value=value)
            )
        elif np.isscalar(value) and np.issubdtype(np.asarray(value).dtype, np.integer):
            value = np.asarray(value).item()
            head.user_parameters.user_parameter_long.append(
                mrd.UserParameterLongType(name=key, value=int(value))
            )
        elif np.issubdtype(np.asarray(value).dtype, np.integer):
            value = np.asarray(value, dtype=int).tolist()
            head.user_parameters.user_parameter_long.append(
                mrd.UserParameterLongType(name=key, value=value)
            )
        elif np.isscalar(value) and np.issubdtype(np.asarray(value).dtype, np.floating):
            value = np.asarray(value).item()
            head.user_parameters.user_parameter_long.append(
                mrd.UserParameterDoubleType(name=key, value=float(value))
            )
        elif np.issubdtype(np.asarray(value).dtype, np.floating):
            value = np.asarray(value, dtype=float).tolist()
            head.user_parameters.user_parameter_long.append(
                mrd.UserParameterDoubleType(name=key, value=value)
            )
        elif isinstance(value, np.ndarray):
            value = serialize_array(value)
            head.user_parameters.user_parameter_base64.append(
                mrd.UserParameterBase64Type(name=key, value=value)
            )
        else:
            try:
                value = base64.b64encode(value).decode("utf-8")
                head.user_parameters.user_parameter_base64.append(
                    mrd.UserParameterBase64Type(name=key, value=value)
                )
            except Exception:
                pass

    # rf phase
    rf_phase = np.atleast_1d(meta.method.VariablePhase).tolist()
    head.user_parameters.user_parameter_double.append(
        mrd.UserParameterDoubleType(name="RFPhase", value=rf_phase)
    )

    # create acquisitions
    ndims = meta.traj.shape[-1]
    npts = meta.traj.shape[-2]
    nviews = meta.traj.shape[-3]
    nslices = enc.encoded_space.matrix_size.z
    ncontrasts = meta.traj.shape[-4]

    if "ReadoutLength" in meta.user:
        ncontrasts = len(np.unique(meta.method.VariableTE))
        trajectory = np.repeat(meta.traj, ncontrasts, axis=-3)
        dcf = np.repeat(meta.dcf, ncontrasts, axis=-2)
    else:
        trajectory = meta.traj
        dcf = meta.dcf
    dcf, _ = np.broadcast_arrays(dcf, trajectory[..., 0])

    # set up indexes
    view_idx = np.arange(nviews)
    slice_idx = np.arange(nslices)
    contrast_idx = np.arange(ncontrasts)

    if "SeparableMode" in meta.user and meta.user["SeparableMode"] > 0:
        if meta.user["SeparableMode"] == 1:
            view_idx, slice_idx, contrast_idx = np.broadcast_arrays(
                view_idx[:, None, None],
                slice_idx[None, :, None],
                contrast_idx[None, None, :],
            )
            trajectory = trajectory.transpose(2, 0, 1, 3, 4)
            dcf = dcf.transpose(2, 0, 1, 3)

        if meta.user["SeparableMode"] == 2:
            view_idx, contrast_idx, slice_idx = np.broadcast_arrays(
                view_idx[:, None, None],
                contrast_idx[None, :, None],
                slice_idx[None, None, :],
            )
            trajectory = trajectory.transpose(2, 1, 0, 3, 4)
            dcf = dcf.transpose(2, 1, 0, 3)

        if meta.user["SeparableMode"] == 3:
            slice_idx, contrast_idx, view_idx = np.broadcast_arrays(
                slice_idx[:, None, None],
                contrast_idx[None, :, None],
                view_idx[None, None, :],
            )
            trajectory = trajectory.transpose(1, 0, 2, 3, 4)
            dcf = dcf.transpose(1, 0, 2, 3)

        if meta.user["SeparableMode"] == 4:
            slice_idx, view_idx, contrast_idx = np.broadcast_arrays(
                slice_idx[:, None, None],
                view_idx[None, :, None],
                contrast_idx[None, None, :],
            )
            trajectory = trajectory.transpose(1, 2, 0, 3, 4)
            dcf = dcf.transpose(1, 2, 0, 3)

        if meta.user["SeparableMode"] == 5:
            contrast_idx, slice_idx, view_idx = np.broadcast_arrays(
                contrast_idx[:, None, None],
                slice_idx[None, :, None],
                view_idx[None, None, :],
            )

        if meta.user["SeparableMode"] == 6:
            contrast_idx, view_idx, slice_idx = np.broadcast_arrays(
                contrast_idx[:, None, None],
                view_idx[None, :, None],
                slice_idx[None, None, :],
            )
            trajectory = trajectory.transpose(0, 2, 1, 3, 4)
            dcf = dcf.transpose(0, 2, 1, 3)

        view_idx = view_idx.ravel()
        slice_idx = slice_idx.ravel()
        contrast_idx = contrast_idx.ravel()

    else:
        view_idx, contrast_idx = np.broadcast_arrays(
            view_idx[:, None],
            contrast_idx[None, :],
        )
        trajectory = trajectory.transpose(1, 0, 2, 3)
        dcf = dcf.transpose(1, 0, 2)

        view_idx = view_idx.ravel()
        slice_idx = np.zeros_like(view_idx)
        contrast_idx = contrast_idx.ravel()

    # initialize scan counter
    nscans = view_idx.size
    scan_counter = np.arange(nscans)

    # flatten trajectory
    trajectory = trajectory.reshape(-1, *trajectory.shape[-2:])

    # get DCF
    dcf = dcf.reshape(-1, meta.dcf.shape[-1])
    dcf = dcf[..., None]

    # append dcf as last trajectory dim
    trajectory = np.concatenate((trajectory, dcf), axis=-1)

    # get sampling time
    sampling_time = np.unique(np.round(np.diff(meta.t), 4)).item(()) * 1e3

    # actual acquisition creation
    acquisitions = []
    defs = mrd.AcquisitionFlags
    for n in range(nscans):
        acq = mrd.Acquisition()

        # encoding counter
        idx = mrd.EncodingCounters()
        idx.kspace_encode_step_1 = view_idx[n]
        if ndims == 3:
            idx.kspace_encode_step_2 = slice_idx[n]
            idx.slice = 0
        else:
            idx.kspace_encode_step_2 = 0
            idx.slice = slice_idx[n]
        idx.contrast = contrast_idx[n]
        acq.head.idx = idx
        acq.head.scan_counter = scan_counter[n]
        acq.head.discard_pre = meta.adc[0]
        acq.head.discard_post = meta.adc[1]
        acq.head.center_sample = npts // 2
        acq.head.sample_time_us = sampling_time

        # flags
        if idx.kspace_encode_step_1 == 0:
            acq.head.flags = defs.FIRST_IN_ENCODE_STEP_1
        if idx.kspace_encode_step_1 == nviews - 1:
            acq.head.flags = defs.LAST_IN_ENCODE_STEP_1
        if ndims == 3:
            if idx.kspace_encode_step_2 == 0:
                acq.head.flags = defs.FIRST_IN_ENCODE_STEP_2
            if idx.kspace_encode_step_2 == nslices - 1:
                acq.head.flags = defs.LAST_IN_ENCODE_STEP_2
        else:
            if idx.slice == 0:
                acq.head.flags = defs.FIRST_IN_SLICE
            if idx.slice == nslices - 1:
                acq.head.flags = defs.LAST_IN_SLICE
        if idx.contrast == 0:
            acq.head.flags = defs.FIRST_IN_CONTRAST
        if idx.contrast == ncontrasts - 1:
            acq.head.flags = defs.LAST_IN_CONTRAST
        if acq.head.scan_counter == nscans - 1:
            acq.head.flags = defs.LAST_IN_MEASUREMENT

        acq.trajectory = trajectory[n].T

        acquisitions.append(acq)

    return acquisitions, head
