"""Image unsorting subroutines."""

__all__ = ["unsort_kspace"]

import numpy as np

import mrd

from ...utils import get_user_param


def unsort_kspace(
    recon_buffers: mrd.ReconBuffer | list[mrd.ReconBuffer],
) -> list[mrd.Acquisition]:
    """
    Unsort input MRD ReconBuffer(s) into a set of MRD Acquisition.

    This latter format is more suitable for I/O operations.

    Parameters
    ----------
    recon_buffers: mrd.ReconBuffer | list[mrd.ReconBuffer]
        Input MRD ReconBuffer(s).

    Returns
    -------
    list[mrd.Acquisition]
        Stack of MRD Acquisitions corresponding to input.

    """
    if isinstance(recon_buffers, mrd.ReconBuffer):
        recon_buffers = [recon_buffers]

    # Get total number of scans
    scan_count = []
    for buffer in recon_buffers:
        for head in buffer.headers:
            scan_count.append(head.scan_counter)
    n_scans = max(scan_count) + 1

    # Fill acquisitions
    acquisitions = [None] * n_scans
    for idx in range(len(recon_buffers)):
        data = buffer[idx].data
        trajectory = buffer[idx].trajectory
        density = buffer[idx].density
        headers = buffer[idx].headers

        # Get axis map
        axis_map = axis_maps[idx]

        # Get number of dimensions
        if trajectory.size > 0:
            n_dims = trajectory.shape[-1]
        else:
            n_dims = get_user_param(head, "ImagingMode", 2)

        # Search singleton
        if n_dims == 2:
            axis_map_keys = [
                "channel",
                "phase",
                "contrast",
                "slice",
                "kspace_encoding_step_1",
                "kspace_encoding_step_0",
            ]
        elif n_dims == 3:
            axis_map_keys = [
                "channel",
                "phase",
                "contrast",
                "kspace_encoding_step_2",
                "kspace_encoding_step_1",
                "kspace_encoding_step_0",
            ]
        axis_size_values = [1, 1, 1, 1, 1, 1]
        axis_size = dict(zip(axis_size_keys, axis_size_values))
        for k, v in axis_map.items():
            axis_size[k] = data.shape[v]
        singleton_axis = np.where(np.asarray(list(axis_size.values())) == 1)[0].tolist()
        singleton_axis = tuple(singleton_axis)

        # Unsqueeze
        data = np.expand_dims(data, singleton_axis)
        if trajectory is not None:
            trajectory = np.expand_dims(trajectory, singleton_axis)
        if density is not None:
            density = np.expand_dims(density, singleton_axis)

        # Reformat
        data = data.transpose(1, 2, 3, 4, 0, 5)
        if trajectory is not None:
            if density is None:
                density = np.ones_like(trajectory[..., 0])
            trajectory = np.concatenate((trajectory, density[..., None]), axis=-1).T
        else:
            trajectory = None
        headers = headers.ravel()

        # Get scan count
        scan_idx = [head.scan_counter for head in headers]
        scan_idx = np.asarray(scan_idx)

        # Get phase encoding idx
        enc1_idx = np.asarray([head.idx.kspace_encode_step_1 for head in headers])
        enc1_idx = np.asarray([idx if idx else 0 for idx in enc1_idx])

        # Get slice idx
        slice_idx = np.asarray([head.idx.slice for head in headers])
        slice_idx = np.asarray([idx if idx else 0 for idx in slice_idx])

        # Get partition encoding idx
        enc2_idx = np.asarray([head.idx.kspace_encode_step_2 for head in headers])
        enc2_idx = np.asarray([idx if idx else 0 for idx in enc2_idx])

        if len(np.unique(slice_idx)) > 1 and len(np.unique(enc2_idx)) > 1:
            raise ValueError("Multislab 3D acquisitions not supported.")
        if len(np.unique(enc2_idx)) > len(np.unique(slice_idx)):
            slice_idx = enc2_idx

        # Get contrast idx
        contrast_idx = [head.idx.contrast for head in buffer.headers]
        contrast_idx = np.asarray([idx if idx else 0 for idx in contrast_idx])

        # Get phase (aka, triggered imaging) idx
        phase_idx = [head.idx.phase for head in headers]
        phase_idx = np.asarray([idx if idx else 0 for idx in phase_idx])

        # Get repetition (aka, dynamic imaging) idx
        repetition_idx = [head.idx.repetition for head in headers]
        repetition_idx = np.asarray([idx if idx else 0 for idx in repetition_idx])

        if len(np.unique(phase_idx)) > 1 and len(np.unique(repetition_idx)) > 1:
            raise ValueError("Multiphase dynamic acquisition not supported.")
        if len(np.unique(repetition_idx)) > len(np.unique(phase_idx)):
            phase_idx = repetition_idx

        # Unsort data and trajectory
        for n in range(len(headers)):
            _data = data[phase_idx[n], contrast_idx[n], slice_idx[n], enc1_idx[n]]
            if trajectory is not None:
                _trajectory = trajectory[
                    phase_idx[n], contrast_idx[n], slice_idx[n], enc1_idx[n]
                ]
                _acq = mrd.Acquisition(
                    head=headers[n], data=_data, trajectory=_trajectory
                )
            else:
                _acq = mrd.Acquisition(head=headers[n], data=_data)
            acquisitions[scan_idx[n]] = _acq

    # Filter missing
    acquisitions = [acq for acq in acquisitions if acq is not None]

    return acquisitions
