"""Image sorting subroutines."""

__all__ = ["sort_images"]

import base64
import json

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

    # Get slice idx
    slice_idx = np.asarray([head.slice for head in _headers])
    slice_idx = np.asarray([idx if idx else 0 for idx in slice_idx])

    # Get contrast idx
    contrast_idx = [head.contrast for head in _headers]
    contrast_idx = np.asarray([idx if idx else 0 for idx in contrast_idx])

    # Get phase (aka, triggered imaging) idx
    phase_idx = [head.phase for head in _headers]
    phase_idx = np.asarray([idx if idx else 0 for idx in phase_idx])

    # Get repetition (aka, dynamic imaging) idx
    repetition_idx = [head.repetition for head in _headers]
    repetition_idx = np.asarray([idx if idx else 0 for idx in repetition_idx])

    if len(np.unique(phase_idx)) > 1 and len(np.unique(repetition_idx)) > 1:
        raise ValueError("Multiphase dynamic acquisition not supported.")
    if len(np.unique(repetition_idx)) > len(np.unique(phase_idx)):
        phase_idx = repetition_idx

    # Get encoding size
    n_slices = len(np.unique(slice_idx))
    n_contrasts = len(np.unique(contrast_idx))
    n_phases = len(np.unique(phase_idx))
    n_instances, ny, nx = _data.shape

    # Fill sorted image tensor
    image_types = np.asarray([img.head.image_type.value for img in images])

    # Initialize sorted headers and meta
    headers = np.empty((n_phases, n_contrasts, n_slices), dtype=object)
    meta = np.empty((n_phases, n_contrasts, n_slices), dtype=object)

    # Sort headers and meta
    for n in range(n_instances):
        _headers[n].image_index = n
        _headers[n].image_type = mrd.ImageType.COMPLEX
        headers[phase_idx[n], contrast_idx[n], slice_idx[n]] = _headers[n]
        meta[phase_idx[n], contrast_idx[n], slice_idx[n]] = _meta[n]

    # Real-valued image
    is_complex = False
    if sum(image_types == 1) == n_instances:
        data = np.zeros((n_phases, n_contrasts, n_slices, ny, nx), dtype=np.complex64)
        for n in range(n_instances):
            data[phase_idx[n], contrast_idx[n], slice_idx[n], :, :] = _data[n].astype(
                np.complex64
            )

    elif sum(image_types == 3) == n_instances:
        data = np.zeros((n_phases, n_contrasts, n_slices, ny, nx), dtype=np.complex64)
        for n in range(n_instances):
            data[phase_idx[n], contrast_idx[n], slice_idx[n], :, :] = _data[n].astype(
                np.complex64
            )

    # Complex images, i.e., assume all input images are complex
    elif sum(image_types == 5) > 0:
        is_complex = True
        if sum(image_types == 5) != n_instances:
            raise RuntimeError("Mixing real and complex-valued images")
        data = np.zeros((n_phases, n_contrasts, n_slices, ny, nx), dtype=np.complex64)
        for n in range(n_instances):
            data[phase_idx[n], contrast_idx[n], slice_idx[n], :, :] = _data[n].astype(
                np.complex64
            )

    # Complex images with separate magn/phase or real/imag
    else:
        is_complex = True
        data = [None, None, None, None]
        if sum(image_types == 1) > 0:
            data[0] = np.zeros(
                (n_phases, n_contrasts, n_slices, ny, nx), dtype=np.float32
            )
        if sum(image_types == 2) > 0:
            data[1] = np.zeros(
                (n_phases, n_contrasts, n_slices, ny, nx), dtype=np.float32
            )
        if sum(image_types == 3) > 0:
            data[2] = np.zeros(
                (n_phases, n_contrasts, n_slices, ny, nx), dtype=np.float32
            )
        if sum(image_types == 4) > 0:
            data[3] = np.zeros(
                (n_phases, n_contrasts, n_slices, ny, nx), dtype=np.float32
            )

        for n in range(n_instances):
            idx = image_types[n] - 1
            data[idx][phase_idx[n], contrast_idx[n], slice_idx[n], :, :] = _data[
                n
            ].astype(np.float32)

        # Real + 1j * Imag
        if sum(image_types == 3) > 0 and sum(image_types == 4) > 0:
            data = data[2] + 1j * data[3]
        else:
            if (data[1] > 2 * np.pi).any():
                data = data[0] * np.exp(1j * (2 * np.pi * data[1] / 4095 - np.pi))
            else:
                data = data[0] * np.exp(1j * data[1])

    # Correct phase shift along z for GE systems
    if (
        is_complex
        and head.acquisition_system_information
        and head.acquisition_system_information.system_vendor
    ):
        if "GE" in head.acquisition_system_information.system_vendor.upper():
            data = np.fft.ifft(
                np.fft.fftshift(np.fft.fft(data, axis=-3), axes=-3), axis=-3
            )

    # Add axis map
    axis_map_keys = ["phase", "contrast", "slice", "rows", "columns"]
    axis_map_keys = np.asarray(axis_map_keys)
    singleton_axis = np.asarray(data.shape) == 1
    axis_map_keys = axis_map_keys[np.logical_not(singleton_axis)].tolist()
    axis_map_values = np.arange(len(axis_map_keys)).tolist()
    axis_map = dict(zip(axis_map_keys, axis_map_values))
    axis_map = base64.b64encode(json.dumps(axis_map).encode("utf-8")).decode("utf-8")
    if head.user_parameters is None:
        head.user_parameters = mrd.UserParametersType()
    head.user_parameters.user_parameter_base64.append(
        mrd.UserParameterBase64Type(name="AxisMaps", value=axis_map)
    )

    return (
        mrd.ImageArray(
            data=data.squeeze(), headers=headers.squeeze(), meta=meta.squeeze()
        ),
        head,
    )
