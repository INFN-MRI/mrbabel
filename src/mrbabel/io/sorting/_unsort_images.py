"""Image unsorting subroutines."""

__all__ = ["unsort_images"]

import numpy as np

import mrd

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

    # Fill images list
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

