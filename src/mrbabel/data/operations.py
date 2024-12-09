"""Sorting and Unsorting utilities."""

__all__ = ["sort_images"]

import numpy as np

import mrd


def sort_images(images: list[mrd.Image]) -> mrd.ImageArray:
    """
    Sort input set of MRD Images into a ImageArray.

    This latter format is more suitable for volumetric processing.

    Parameters
    ----------
    images: list[mrd.Images]
        Input stack of MRD Images.

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

    # Real + 1j * Imag
    if sum(image_types == 3) > 0 and sum(image_types == 4) > 0:
        data = data[2] + 1j * data[3]
    else:
        data = data[0] * np.exp(1j * (2 * np.pi * data[1] / 4095 - np.pi))

    return mrd.ImageArray(data=data, headers=_headers, meta=_meta)