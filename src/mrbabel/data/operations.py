"""Sorting and Unsorting utilities."""

__all__ = ["sort_images", "sort_kspace", "unsort_images", "unsort_kspace"]

import numpy as np

import mrd


def sort_images(images: list[mrd.Image], head: mrd.Header) -> mrd.ImageArray:
    """
    Sort input set of MRD Images into a MRD ImageArray.

    This latter format is more suitable for volumetric processing.

    Parameters
    ----------
    images : list[mrd.Images]
        Input stack of MRD Images.
    head : mrd.Header
        MRD Header corresponding to input MRD Images.

    Returns
    -------
    mrd.ImageArray
        Sorted MRD ImageArray built from input.

    """
    _meta = np.asarray([img.meta for img in images])
    _headers = np.asarray([img.head for img in images])
    _data = np.stack([img.data for img in images])

    # get unique contrast and indexes
    contrast_idx = np.asarray([img.head.contrast for img in images])

    # get slice locations
    slice_idx = np.asarray([img.head.slice for img in images])

    # get matrix size
    n_slices = len(np.unique(slice_idx))
    n_contrasts = len(np.unique(contrast_idx))
    n_instances, ny, nx = _data.shape

    # fill sorted image tensor
    image_types = np.asarray([img.head.image_type.value for img in images])

    # Real-valued image
    if sum(image_types == 1) == n_instances:
        data = np.zeros((n_contrasts, n_slices, ny, nx), dtype=np.complex64)
        for n in range(n_instances):
            data[contrast_idx[n], slice_idx[n], :, :] = _data[n].astype(np.complex64)
        return mrd.ImageArray(data=data, headers=_headers, meta=_meta)

    if sum(image_types == 3) == n_instances:
        data = np.zeros((n_contrasts, n_slices, ny, nx), dtype=np.complex64)
        for n in range(n_instances):
            data[contrast_idx[n], slice_idx[n], :, :] = _data[n].astype(np.complex64)
        return mrd.ImageArray(data=data, headers=_headers, meta=_meta)

    # Complex images, i.e., assume all input images are complex
    if sum(image_types == 5) > 0:
        if sum(image_types == 5) != n_instances:
            raise RuntimeError("Mixing real and complex-valued images")
        data = np.zeros((n_contrasts, n_slices, ny, nx), dtype=np.complex64)
        for n in range(n_instances):
            data[contrast_idx[n], slice_idx[n], :, :] = _data[n].astype(np.complex64)
        return mrd.ImageArray(data=data, headers=_headers, meta=_meta)

    # Complex images with separate magn/phase or real/imag
    data = [None, None, None, None]
    if sum(image_types == 1) > 0:
        data[0] = np.zeros((n_contrasts, n_slices, ny, nx), dtype=np.float32)
    if sum(image_types == 2) > 0:
        data[1] = np.zeros((n_contrasts, n_slices, ny, nx), dtype=np.float32)
    if sum(image_types == 3) > 0:
        data[2] = np.zeros((n_contrasts, n_slices, ny, nx), dtype=np.float32)
    if sum(image_types == 4) > 0:
        data[3] = np.zeros((n_contrasts, n_slices, ny, nx), dtype=np.float32)

    for n in range(n_instances):
        idx = image_types[n] - 1
        data[idx][contrast_idx[n], slice_idx[n], :, :] = _data[n].astype(np.float32)

    # Real + 1j * Imag
    if sum(image_types == 3) > 0 and sum(image_types == 4) > 0:
        data = data[2] + 1j * data[3]
    else:
        if (data[1] > 2 * np.pi).any():
            data = data[0] * np.exp(1j * (2 * np.pi * data[1] / 4095 - np.pi))
        else:
            data = data[0] * np.exp(1j * data[1])

    if (
        head.acquisition_system_information
        and head.acquisition_system_information.system_vendor
    ):
        if "GE" in head.acquisition_system_information.system_vendor.upper():
            data = np.fft.ifft(
                np.fft.fftshift(np.fft.fft(data, axis=-3), axes=-3), axis=-3
            )

    return mrd.ImageArray(data=data, headers=_headers, meta=_meta)


def sort_kspace(
    acquisitions: list[mrd.Acquisition], head: mrd.Header
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
    _density = [[]] * n_encoded_spaces
    _trajectory = [[]] * n_encoded_spaces
    _data = [[]] * n_encoded_spaces
    _headers = [[]] * n_encoded_spaces

    # split data, headers and trajectories for the different encodings
    for n in range(len(acquisitions)):
        idx = _encoding_spaces[n]
        if acquisitions[n].trajectory.size > 0:
            _density[idx].append(acquisitions[n].trajectory[..., -1])
            _trajectory[idx].append(acquisitions[n].trajectory[..., :-1])
        _data[idx].append(acquisitions[n].data)
        _headers[idx].append(acquisitions[n].head)

    # loop over encodings
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

        # get contrast idx
        contrast_idx = [head.idx.contrast for head in headers]
        contrast_idx = np.asarray([idx if idx else 0 for idx in contrast_idx])

        # get slice idx
        slice_idx_1 = np.asarray([head.idx.slice for head in headers])
        slice_idx_2 = np.asarray([head.idx.kspace_encode_step_2 for head in headers])

        if len(np.unique(slice_idx_1)) > 1 and len(np.unique(slice_idx_2)) > 1:
            raise ValueError("Multislab 3D acquisitions not supported.")
        slice_idx = slice_idx_2
        if len(np.unique(slice_idx_1)) > len(np.unique(slice_idx_2)):
            slice_idx = slice_idx_1

        # get phase idx
        phase_idx = np.asarray([head.idx.kspace_encode_step_1 for head in headers])

        # get encoding size
        n_coils = data.shape[-2]
        n_pts = data.shape[-1]
        n_phases = len(np.unique(phase_idx))
        n_slices = len(np.unique(slice_idx))
        n_contrasts = len(np.unique(contrast_idx))

        # sort data and trajectory
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
        for idx in range(data.shape[0]):
            buffered_data[contrast_idx[idx], slice_idx[idx], phase_idx[idx]] = data[idx]

        # reshape to (ncoils, n_contrasts, n_slices, n_phases, n_pts)
        buffered_data = buffered_data.transpose(3, 0, 1, 2, 4)

        if trajectory.size > 0:
            n_dims = trajectory.shape[-1]
            buffered_trajectory = np.zeros(
                (
                    n_contrasts,
                    n_slices,
                    n_phases,
                    n_pts,
                    n_dims,
                ),
                dtype=np.complex64,
            )
            buffered_density = np.zeros(
                (
                    n_contrasts,
                    n_slices,
                    n_phases,
                    n_pts,
                ),
                dtype=np.complex64,
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

        # prepare sampling description
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
            data=buffered_data,
            trajectory=buffered_trajectory,
            density=buffered_density,
            headers=headers,
            sampling=sampling,
        )
        recon_buffers.append(buffer)

    if len(recon_buffers) == 1:
        recon_buffers = recon_buffers[0]

    return recon_buffers


def unsort_images(image: mrd.ImageArray) -> list[mrd.Image]:
    """
    Unsort input MRD ImageArray into a set of MRD Images.

    This latter format is more suitable for I/O operations.

    Parameters
    ----------
    image : mrd.ImageArray
        Input MRD ImageArray.

    Returns
    -------
    list[mrd.Images]
        Stack of MRD Images corresponding to input.

    """
    _data = image.data
    _headers = image.headers
    _meta = image.meta

    # fill images list
    images = []
    n_images = len(_headers)
    for n in range(n_images):
        imtype = _headers[n].image_type
        contrast_idx = _headers[n].contrast
        slice_idx = _headers[n].slice
        if imtype == mrd.ImageType.COMPLEX:
            data = _data[contrast_idx, slice_idx].squeeze()
        if imtype == mrd.ImageType.MAGNITUDE:
            data = np.abs(_data[contrast_idx, slice_idx]).squeeze()
        if imtype == mrd.ImageType.PHASE:
            data = np.angle(_data[contrast_idx, slice_idx]).squeeze()
        if imtype == mrd.ImageType.REAL:
            data = _data[contrast_idx, slice_idx].squeeze().real
        if imtype == mrd.ImageType.IMAG:
            data = _data[contrast_idx, slice_idx].squeeze().imag
        images.append(mrd.Image(data=data, head=_headers[n], meta=_meta[n]))

    return images


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

    # get total number of scans
    scan_count = []
    for buffer in recon_buffers:
        for head in buffer.headers:
            scan_count.append(head.scan_counter)
    n_scans = max(scan_count) + 1

    # fill acquisitions
    acquisitions = [None] * n_scans
    for buffer in recon_buffers:
        data = buffer.data.transpose(1, 2, 3, 0, 4)
        if buffer.trajectory is not None and buffer.density is not None:
            trajectory = np.concatenate(
                (buffer.trajectory, buffer.density[..., None]), axis=-1
            )
        else:
            trajectory = None

        # get scan count
        scan_idx = [head.scan_counter for head in buffer.headers]
        scan_idx = np.asarray(scan_idx)

        # get contrast idx
        contrast_idx = [head.idx.contrast for head in buffer.headers]
        contrast_idx = np.asarray([idx if idx else 0 for idx in contrast_idx])

        # get slice idx
        slice_idx_1 = np.asarray([head.idx.slice for head in buffer.headers])
        slice_idx_2 = np.asarray(
            [head.idx.kspace_encode_step_2 for head in buffer.headers]
        )

        if len(np.unique(slice_idx_1)) > 1 and len(np.unique(slice_idx_2)) > 1:
            raise ValueError("Multislab 3D acquisitions not supported.")
        slice_idx = slice_idx_2
        if len(np.unique(slice_idx_1)) > len(np.unique(slice_idx_2)):
            slice_idx = slice_idx_1

        # get phase idx
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

    # filter missing
    acquisitions = [acq for acq in acquisitions if acq is not None]

    return acquisitions
