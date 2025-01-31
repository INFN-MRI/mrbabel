"""Image unsorting subroutines."""

__all__ = ["unsort_images"]

import copy

import numpy as np

import mrd

from ...utils import get_user_param


def unsort_images(image: mrd.ImageArray, head: mrd.Header) -> list[mrd.Image]:
    """
    Unsort input MRD ImageArray into a set of MRD Images.

    This latter format is more suitable for I/O operations.

    Parameters
    ----------
    image : mrd.ImageArray
        Input MRD ImageArray.
    head: mrd.Header
        Input MRD Header.

    Returns
    -------
    list[mrd.Images]
        Stack of MRD Images corresponding to input.

    """
    _data = image.data
    _headers = image.headers
    _meta = image.meta

    # Get axis map
    axis_map = get_user_param(head, "AxisMaps")

    # Search singleton
    axis_size_keys = ["phase", "contrast", "slice", "rows", "columns"]
    axis_size_values = [1, 1, 1, 1, 1]
    axis_size = dict(zip(axis_size_keys, axis_size_values))
    for k, v in axis_map.items():
        axis_size[k] = _data.shape[v]
    singleton_axis = np.where(np.asarray(list(axis_size.values())) == 1)[0].tolist()
    singleton_axis = tuple(singleton_axis)

    # Unsqueeze
    _data = np.expand_dims(_data, singleton_axis)

    # Reformat
    _headers = _headers.ravel()
    _meta = _meta.ravel()

    # Fill images list
    images = []
    n_images = len(_headers)
    counter = 0
    for n in range(n_images):
        phase_idx = _get_phase_idx(_headers[n])
        contrast_idx = _headers[n].contrast
        slice_idx = _headers[n].slice

        # Update
        if _headers[n].image_type == mrd.ImageType.COMPLEX:
            # if imtype == mrd.ImageType.MAGNITUDE:
            # Magnitude
            image_header = copy.deepcopy(_headers[n])
            image_header.image_type = mrd.ImageType.MAGNITUDE
            image_header.image_index = counter
            data = np.abs(_data[phase_idx, contrast_idx, slice_idx].squeeze())
            images.append(mrd.Image(data=data, head=image_header, meta=_meta[n]))
            counter += 1

            # Real
            image_header = copy.deepcopy(_headers[n])
            image_header.image_type = mrd.ImageType.REAL
            image_header.image_index = counter
            data = (_data[phase_idx, contrast_idx, slice_idx].squeeze()).real
            images.append(mrd.Image(data=data, head=image_header, meta=_meta[n]))
            counter += 1

            # Imaginary
            image_header = copy.deepcopy(_headers[n])
            image_header.image_type = mrd.ImageType.IMAG
            image_header.image_index = counter
            data = (_data[phase_idx, contrast_idx, slice_idx].squeeze()).imag
            images.append(mrd.Image(data=data, head=image_header, meta=_meta[n]))
            counter += 1
        else:
            image_header = _headers[n]
            image_header.image_index = counter
            data = _data[phase_idx, contrast_idx, slice_idx].squeeze()
            images.append(mrd.Image(data=data, head=image_header, meta=_meta[n]))
            counter += 1

    return images


def _get_phase_idx(image_header):  # noqa
    phase_idx = image_header.phase
    if phase_idx is None:
        phase_idx = 0
    repetition_idx = image_header.repetition
    if repetition_idx is None:
        repetition_idx = 0
    return max(phase_idx, repetition_idx)
