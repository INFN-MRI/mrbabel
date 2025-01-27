"""K-space sorting subroutines."""

__all__ = ["sort_kspace"]

import numpy as np

import mrd

from ...utils import get_user_param

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
        phase_idx = np.asarray([head.idx.kspace_encode_step_1 for head in headers])
        
        # Get slice idx
        slice_idx_1 = np.asarray([head.idx.slice for head in headers])
        slice_idx_2 = np.asarray([head.idx.kspace_encode_step_2 for head in headers])

        if len(np.unique(slice_idx_1)) > 1 and len(np.unique(slice_idx_2)) > 1:
            raise ValueError("Multislab 3D acquisitions not supported.")
        slice_idx = slice_idx_2
        if len(np.unique(slice_idx_1)) > len(np.unique(slice_idx_2)):
            slice_idx = slice_idx_1

        # Get contrast idx
        contrast_idx = [head.idx.contrast for head in headers]
        contrast_idx = np.asarray([idx if idx else 0 for idx in contrast_idx])
                                        
        # Get encoding size
        n_pts = data.shape[-1]
        n_coils = data.shape[-2]
        n_phases = len(np.unique(phase_idx))
        n_slices = len(np.unique(slice_idx))
        n_contrasts = len(np.unique(contrast_idx))
        
        # Sort data and trajectory
        buffered_data = np.zeros(
            (
                n_contrasts,
                n_slices,
                n_phases,
                n_coils,
                n_pts,
            ),
            dtype=np.complex64,
        )
        buffered_headers = np.empty(
            (
                n_contrasts,
                n_slices,
                n_phases,
            ),
            dtype=object,
        )
        for idx in range(data.shape[0]):
            buffered_data[contrast_idx[idx], slice_idx[idx], phase_idx[idx]] = data[idx]
            buffered_headers[contrast_idx[idx], slice_idx[idx], phase_idx[idx]] = headers[idx]

        # Reshape to (ncoils, n_contrasts, n_slices, n_phases, n_pts)
        buffered_data = buffered_data.transpose(3, 0, 1, 2, 4)

        if trajectory.size > 0:
            n_dims = trajectory.shape[-1]
            n_pts = trajectory.shape[-2]
            buffered_trajectory = np.zeros(
                (
                    n_contrasts,
                    n_slices,
                    n_phases,
                    n_pts,
                    n_dims,
                ),
                dtype=np.float32,
            )
            buffered_density = np.zeros(
                (
                    n_contrasts,
                    n_slices,
                    n_phases,
                    n_pts,
                ),
                dtype=np.float32,
            )
            for idx in range(data.shape[0]):
                buffered_trajectory[
                    contrast_idx[idx], slice_idx[idx], phase_idx[idx]
                ] = trajectory[idx]
                buffered_density[contrast_idx[idx], slice_idx[idx], phase_idx[idx]] = (
                    density[idx]
                )
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

        sampling.sampling_limits.kspace_encoding_step_0.maximum = n_pts
        sampling.sampling_limits.kspace_encoding_step_0.center = n_pts // 2
        sampling.sampling_limits.kspace_encoding_step_1.maximum = n_phases
        sampling.sampling_limits.kspace_encoding_step_1.center = n_phases // 2
        sampling.sampling_limits.kspace_encoding_step_2.maximum = n_slices
        sampling.sampling_limits.kspace_encoding_step_2.center = n_slices // 2

        buffer = mrd.ReconBuffer(
            data=np.ascontiguousarray(buffered_data),
            trajectory=np.ascontiguousarray(buffered_trajectory),
            density=np.ascontiguousarray(buffered_density),
            headers=buffered_headers,
            sampling=sampling,
        )
        recon_buffers.append(buffer)

    if len(recon_buffers) == 1:
        recon_buffers = recon_buffers[0]

    return recon_buffers
