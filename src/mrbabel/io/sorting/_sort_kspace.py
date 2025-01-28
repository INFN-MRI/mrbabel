"""K-space sorting subroutines."""

__all__ = ["sort_kspace"]

import base64
import json

import numpy as np

import mrd


def sort_kspace(
    acquisitions: list[mrd.Acquisition],
    head: mrd.Header,
) -> mrd.ReconBuffer | list[mrd.ReconBuffer]:
    """
    Sort input MRD Acquisitions into MRD ReconBuffer.

    This latter format is more suitable for reconstruction.

    Parameters
    ----------
    acquisitions : list[mrd.Acquisition]
        Input MRD Acquisitions.
    head : mrd.Header
        MRD Header corresponding to input MRD Acquisitions.

    Returns
    -------
    recon_buffers : mrd.ReconBuffer | list[mrd.ReconBuffer]
        Sorted ReconBuffer for each encoded space.

    """
    _encoding_spaces = [acq.head.encoding_space_ref for acq in acquisitions]
    _encoding_spaces = np.asarray([idx if idx else 0 for idx in _encoding_spaces])

    # number of encoded spaces
    n_encoded_spaces = len(np.unique(_encoding_spaces))
    _density = [[] for n in range(n_encoded_spaces)]
    _trajectory = [[] for n in range(n_encoded_spaces)]
    _data = [[] for n in range(n_encoded_spaces)]
    _headers = [[] for n in range(n_encoded_spaces)]

    # Split data, headers and trajectories for the different encodings
    for n in range(len(acquisitions)):
        idx = _encoding_spaces[n]
        if acquisitions[n].trajectory.size > 0:
            _density[idx].append(acquisitions[n].trajectory.T[..., -1])
            _trajectory[idx].append(acquisitions[n].trajectory.T[..., :-1])
        _data[idx].append(acquisitions[n].data)
        _headers[idx].append(acquisitions[n].head)

    # Loop over encodings
    recon_buffers = []
    axis_maps = []
    for n in range(n_encoded_spaces):
        data = np.stack([d for d in _data[n]])
        try:
            trajectory = np.stack([traj for traj in _trajectory[n]])
            density = np.stack([dens for dens in _density[n]])
        except Exception:
            trajectory = np.asarray([])
            density = np.asarray([])
        headers = _headers[n]

        # Get phase idx
        enc1_idx = np.asarray([head.idx.kspace_encode_step_1 for head in headers])

        # Get slice idx
        ndim = 2
        slice_idx = np.asarray([head.idx.slice for head in headers])
        enc2_idx = np.asarray([head.idx.kspace_encode_step_2 for head in headers])

        if len(np.unique(slice_idx)) > 1 and len(np.unique(enc2_idx)) > 1:
            raise ValueError("Multislab 3D acquisitions not supported.")
        if len(np.unique(enc2_idx)) > len(np.unique(slice_idx)):
            slice_idx = enc2_idx
            ndim = 3

        # Get contrast idx
        contrast_idx = [head.idx.contrast for head in headers]
        contrast_idx = np.asarray([idx if idx else 0 for idx in contrast_idx])

        # Get phase/frame (aka, dynamic imaging) idx
        phase_idx = [head.idx.phase for head in headers]
        phase_idx = np.asarray([idx if idx else 0 for idx in phase_idx])

        # Get average idx
        average_idx = [head.idx.average for head in headers]
        average_idx = np.asarray([idx if idx else 0 for idx in average_idx])

        # Get encoding size
        n_channels = data.shape[-2]
        n_samples = data.shape[-1]
        n_readouts = len(np.unique(enc1_idx))
        n_slices = len(np.unique(slice_idx))
        n_contrasts = len(np.unique(contrast_idx))
        n_phases = len(np.unique(phase_idx))
        n_averages = len(np.unique(average_idx))

        # Sort data and trajectory
        buffered_data = np.zeros(
            (
                n_averages,
                n_phases,
                n_contrasts,
                n_slices,
                n_readouts,
                n_channels,
                n_samples,
            ),
            dtype=np.complex64,
        )
        buffered_headers = np.empty(
            (
                n_phases,
                n_contrasts,
                n_slices,
                n_readouts,
            ),
            dtype=object,
        )
        for idx in range(data.shape[0]):
            buffered_data[
                average_idx[idx],
                phase_idx[idx],
                contrast_idx[idx],
                slice_idx[idx],
                enc1_idx[idx],
            ] = data[idx]
            buffered_headers[
                phase_idx[idx], contrast_idx[idx], slice_idx[idx], enc1_idx[idx]
            ] = headers[idx]

        # Reshape to (n_averages, n_channels, n_phases, n_contrasts, n_slices, n_readouts, n_samples)
        buffered_data = buffered_data.transpose(0, 5, 1, 2, 3, 4, 6)
        buffered_data = buffered_data.mean(axis=0)  # average data

        if trajectory.size > 0:
            n_dims = trajectory.shape[-1]
            n_samples = trajectory.shape[-2]
            buffered_trajectory = np.zeros(
                (
                    n_phases,
                    n_contrasts,
                    n_slices,
                    n_readouts,
                    n_samples,
                    n_dims,
                ),
                dtype=np.float32,
            )
            buffered_density = np.zeros(
                (
                    n_phases,
                    n_contrasts,
                    n_slices,
                    n_readouts,
                    n_samples,
                ),
                dtype=np.float32,
            )
            for idx in range(data.shape[0]):
                buffered_trajectory[
                    phase_idx[idx], contrast_idx[idx], slice_idx[idx], enc1_idx[idx]
                ] = trajectory[idx]
                buffered_density[
                    phase_idx[idx], contrast_idx[idx], slice_idx[idx], enc1_idx[idx]
                ] = density[idx]
        else:
            buffered_trajectory = None
            buffered_density = None

        # Prepare sampling description
        sampling = mrd.SamplingDescription()
        sampling.encoded_fov.x = head.encoding[n].encoded_space.field_of_view_mm.x
        sampling.encoded_fov.y = head.encoding[n].encoded_space.field_of_view_mm.y
        sampling.encoded_fov.z = head.encoding[n].encoded_space.field_of_view_mm.z

        sampling.encoded_matrix.x = head.encoding[n].encoded_space.matrix_size.x
        sampling.encoded_matrix.y = head.encoding[n].encoded_space.matrix_size.y
        sampling.encoded_matrix.z = head.encoding[n].encoded_space.matrix_size.z

        sampling.recon_fov.x = head.encoding[n].recon_space.field_of_view_mm.x
        sampling.recon_fov.y = head.encoding[n].recon_space.field_of_view_mm.y
        sampling.recon_fov.z = head.encoding[n].recon_space.field_of_view_mm.z

        sampling.recon_matrix.x = head.encoding[n].recon_space.matrix_size.x
        sampling.recon_matrix.y = head.encoding[n].recon_space.matrix_size.y
        sampling.recon_matrix.z = head.encoding[n].recon_space.matrix_size.z

        sampling.sampling_limits.kspace_encoding_step_0.maximum = n_samples
        sampling.sampling_limits.kspace_encoding_step_0.center = n_samples // 2
        sampling.sampling_limits.kspace_encoding_step_1.maximum = n_readouts
        sampling.sampling_limits.kspace_encoding_step_1.center = n_readouts // 2
        if ndim == 3:
            sampling.sampling_limits.kspace_encoding_step_2.maximum = n_slices
            sampling.sampling_limits.kspace_encoding_step_2.center = n_slices // 2

        buffer = mrd.ReconBuffer(
            data=np.ascontiguousarray(buffered_data),
            trajectory=np.ascontiguousarray(buffered_trajectory),
            density=np.ascontiguousarray(buffered_density),
            headers=buffered_headers,
            sampling=sampling,
        )

        # add axis map
        if ndim == 2:
            axis_map_keys = [
                "channel",
                "phase",
                "contrast",
                "slice",
                "kspace_encoding_step_1",
                "kspace_encoding_step_0",
            ]
        elif ndim == 3:
            axis_map_keys = [
                "channel",
                "phase",
                "contrast",
                "kspace_encoding_step_2",
                "kspace_encoding_step_1",
                "kspace_encoding_step_0",
            ]
        axis_map_keys = np.asarray(axis_map_keys)
        singleton_axis = np.asarray(buffer.data.shape) == 1
        axis_map_keys = axis_map_keys[np.logical_not(singleton_axis)].tolist()
        axis_map_values = np.arange(len(axis_map_keys)).tolist()
        axis_map = dict(zip(axis_map_keys, axis_map_values))
        axis_maps.append(axis_map)

        # append buffer
        buffer.data = buffer.data.squeeze()
        buffer.headers = buffer.headers.squeeze()
        if trajectory.size > 0:
            buffer.trajectory = buffer.trajectory.squeeze()
            buffer.density = buffer.density.squeeze()
        recon_buffers.append(buffer)

    if len(recon_buffers) == 1:
        recon_buffers = recon_buffers[0]
        axis_maps = axis_maps[0]

    axis_maps = base64.b64encode(json.dumps(axis_maps).encode("utf-8")).decode("utf-8")
    if head.user_parameters is None:
        head.user_parameters = mrd.UserParametersType()
    head.user_parameters.user_parameter_base64.append(
        mrd.UserParameterBase64Type(name="AxisMaps", value=axis_maps)
    )

    return recon_buffers, head
