"""FOV centering routines."""

__all__ = ["shift_buffer", "shift_kspace"]


import numpy as np
from numpy.typing import NDArray

import mrd

from ._user import get_user_param


def shift_buffer(input: mrd.ReconBuffer, head: mrd.Header) -> mrd.ReconBuffer:
    """
    Shift input MRD ReconBuffer.

    Parameters
    ----------
    input : mrd.ReconBuffer
        Input MRD ReconBuffer.
    head : mrd.Header
        MRD Header corresponding to input MRD ReconBuffer.

    Returns
    -------
    mrd.ReconBuffer
        Shift MRD ReconBuffer.

    """
    input_data = input.data
    shift = input.headers.ravel()[0].position  # in mm

    # get matrix size
    nz = head.encoding[-1].encoded_space.matrix_size.z
    ny = head.encoding[-1].encoded_space.matrix_size.y
    nx = head.encoding[-1].encoded_space.matrix_size.x
    shape = (nx, ny, nz)

    # get field of view in mm
    fov_z = head.encoding[-1].encoded_space.field_of_view_mm.z
    fov_y = head.encoding[-1].encoded_space.field_of_view_mm.y
    fov_x = head.encoding[-1].encoded_space.field_of_view_mm.x
    fov = (fov_x, fov_y, fov_z)

    # get resolution in mm
    res = np.asarray(fov, dtype=np.float32) / np.asarray(shape, dtype=np.float32)

    # convert shift from mm to voxel units
    shift = -np.asarray(shift, dtype=np.float32) / res

    # get coordinates
    coords = input.trajectory
    if coords is None:  # Cartesian
        ndim = get_user_param(head, "ImagingMode", "3D")
        if "2" in ndim:
            ndim = 2
        elif "3" in ndim:
            ndim = 3
        else:
            raise ValueError("Number of spatial encoding dimensions not recognized")
        output_data = shift_kspace(input_data, shift[:ndim])
    else:
        ndim = coords.shape[-1]
        shape = shape[:ndim]
        output_data = shift_kspace(input_data, shift[:ndim], coords, shape[::-1])

    # replace shifted data
    output = input
    output.data = output_data
    return output


def shift_kspace(
    input: NDArray[complex],
    shift: tuple[float] | list[float] | NDArray[float],
    coords: NDArray[float] | None = None,
    shape: tuple[int] | list[int] | None = None,
) -> NDArray[complex]:
    """
    Shift input k-space.

    Parameters
    ----------
    input : NDArray[complex]
        Input Fourier space data.
    shift : tuple[float] | list[float] | NDArray[float]
        Shift in voxel units ``(dx, dy, dz)`` or ``(dx, dy)``.
    coords : NDArray[float] | None, optional
        Fourier space coordinates.
        The default is ``None`` (assume rectilinear grid).
    shape: tuple[int] | list[int] | None, optional
        Matrix size ``(nz, ny, nx)``.
        The default is ``None`` (estimate from data).

    Returns
    -------
    NDArray[complex]
        Shifted Fourier space data.

    """
    ndim = len(shift)

    # process Fourier space coordinates
    if coords is None:  # generate Cartesian grid if required
        shape = input.shape[-ndim:]
        coords = np.meshgrid(
            *[
                np.linspace(-shape[n] // 2, shape[n] // 2 - 1, shape[n])
                for n in range(ndim)
            ],
            indexing="ij",
        )
        coords = coords[::-1]  # from (z, y, x) to (x, y, z)
        coords = np.stack(coords, axis=-1).astype(np.float32)
        shape = np.asarray(shape[::-1], dtype=np.float32)
    else:
        ndim = coords.shape[-1]
        if shape is None:
            shape = 2 * np.asarray(
                [np.ceil(np.max(abs(coords[..., n]))) for n in range(ndim)],
                dtype=np.float32,
            )  # (x, y, z)
        else:
            shape = np.asarray(shape[::-1], dtype=np.float32)  # (x, y, z)

    # make sure coords are between (-0.5, 0.5)
    scale = 2 * abs(coords.reshape(-1, ndim)).max(axis=0)
    coords = coords.copy() / scale
    coords = coords.astype(np.float32)

    # calculate linear phase corresponding to given shift
    # (i.e. Fourier Shift Theorem)
    shift = np.asarray(shift, dtype=np.float32)
    arg = (coords * shift).sum(axis=-1)
    phase = np.exp(1j * 2 * np.pi * arg)

    # apply shift
    output = input * phase

    return output
