"""Image unsorting subroutines."""

__all__ = ["unsort_kspace"]

import numpy as np

import mrd

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
    for buffer in recon_buffers:
        data = buffer.data.transpose(1, 2, 3, 0, 4)
        if buffer.trajectory is not None and buffer.density is not None:
            trajectory = np.concatenate(
                (buffer.trajectory, buffer.density[..., None]), axis=-1
            ).T
        else:
            trajectory = None

        # Get scan count
        scan_idx = [head.scan_counter for head in buffer.headers]
        scan_idx = np.asarray(scan_idx)

        # Get contrast idx
        contrast_idx = [head.idx.contrast for head in buffer.headers]
        contrast_idx = np.asarray([idx if idx else 0 for idx in contrast_idx])

        # Get slice idx
        slice_idx_1 = np.asarray([head.idx.slice for head in buffer.headers])
        slice_idx_2 = np.asarray(
            [head.idx.kspace_encode_step_2 for head in buffer.headers]
        )

        if len(np.unique(slice_idx_1)) > 1 and len(np.unique(slice_idx_2)) > 1:
            raise ValueError("Multislab 3D acquisitions not supported.")
        slice_idx = slice_idx_2
        if len(np.unique(slice_idx_1)) > len(np.unique(slice_idx_2)):
            slice_idx = slice_idx_1

        # Get phase idx
        phase_idx = np.asarray(
            [head.idx.kspace_encode_step_1 for head in buffer.headers]
        )
        for n in range(len(buffer.headers)):
            _data = data[contrast_idx[n], slice_idx[n], phase_idx[n]]
            if trajectory is not None:
                _trajectory = trajectory[contrast_idx, slice_idx, phase_idx]
                _acq = mrd.Acquisition(
                    head=buffer.headers[n], data=_data, trajectory=_trajectory
                )
            else:
                _acq = mrd.Acquisition(head=buffer.headers[n], data=_data)
            acquisitions[scan_idx[n]] = _acq

    # Filter missing
    acquisitions = [acq for acq in acquisitions if acq is not None]

    return acquisitions
