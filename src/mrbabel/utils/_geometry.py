"""Geometry estimation tools.

Adapted from https://github.com/pehses/twixtools/blob/master/twixtools/geometry.py
"""

__all__ = ["get_geometry"]

import numpy as np
import nibabel as nib

from nibabel.orientations import axcodes2ornt, ornt_transform

import mrd

from ._user import get_user_param


def get_geometry(
    head: mrd.Header,
    raw_or_image_headers: list[mrd.AcquisitionHeader] | list[mrd.ImageHeader],
) -> "Geometry":
    """
    Get geometry info from MRD Header.

    Parameters
    ----------
    head : mrd.Header
        Input MRD Header.
    raw_or_image_headers : list[mrd.AcquisitionHeader] | list[mrd.ImageHeader]
        Either AcquisitionHeader(s) or ImageHeader(s).

    Returns
    -------
    Geometry
        Acquisition geometric info.

    """
    return Geometry(head, raw_or_image_headers)


# %% local utils
class Geometry:
    """Get geometric information from MRD Header."""

    def __repr__(self):  # noqa
        if self.patient_position:
            pp = "".join(self.patient_position.name.split("_"))
        else:
            pp = None
        return (
            "Geomatry info:\n"
            f"  inplane_rot: {np.round(self.inplane_rot, 4)}\n"
            f"  normal: {np.round(self.normal, 4)}\n"
            f"  offset: {np.round(self.offset, 4)}\n"
            f"  patient_position: {pp}\n"
            f"  rotmatrix: {np.round(self.rotmatrix, 4)}\n"
            f"  voxelsize: {np.round(self.voxelsize, 4)}"
        )

    def __str__(self):  # noqa
        if self.patient_position:
            pp = "".join(self.patient_position.name.split("_"))
        else:
            pp = None
        return (
            "Geomatry info:\n"
            f"  inplane_rot: {self.inplane_rot}\n"
            f"  normal: {self.normal}\n"
            f"  offset: {self.offset}\n"
            f"  patient_position: {pp}\n"
            f"  rotmatrix: {self.rotmatrix}\n"
            f"  voxelsize: {self.voxelsize}"
        )

    def __init__(self, head, raw_or_image_headers):  # noqa
        if get_user_param(head, "ImagingMode"):
            self.dims = int(get_user_param(head, "ImagingMode")[0])
        else:
            self.dims = None
        self.fov = get_fov(head)
        self.shape = get_shape(head)
        self.voxelsize = self.get_resolution(head)

        # get normal
        positions = get_image_position(head, raw_or_image_headers)
        orientation = get_image_orientation(head, raw_or_image_headers)[:, 0].reshape(
            2, 3
        )
        self.normal = get_plane_normal(orientation).tolist()

        # get position
        self.offset = positions.mean(axis=-1).tolist()

        # get patient position
        self.patient_position = None
        if head.measurement_information:
            if head.measurement_information.patient_position.name is not None:
                self.patient_position = head.measurement_information.patient_position

        # create affine
        try:
            affine = make_nifti_affine(self.shape, positions, orientation, self.voxelsize)
        except Exception:
            affine = None

        # flip fov, shape, res to (x,y,z)
        self.fov = self.fov[::-1]
        self.shape = self.shape[::-1]
        self.voxelsize = self.voxelsize[::-1]

        # get rotation and in-plane rotation angle
        self.rotmatrix = extract_rotation_from_affine(affine)
        self.inplane_rot = extract_inplane_rotation(self.rotmatrix)

        # reorient affine
        self.affine = affine
        try:
            if self.dims and self.dims == 2:
                scan_orient = detect_scan_orientation(orientation)
                # Axial
                if scan_orient == "ax":
                    self.affine = reorient_affine(affine, self.shape, "RPS")
                # Coronal
                if scan_orient == "cor":
                    self.affine = reorient_affine(affine, self.shape, "RSA")
                # Sagittal
                if scan_orient == "sag":
                    self.affine = reorient_affine(affine, self.shape, "ASR")
                self.affine[:2] *= -1
            if self.dims and self.dims == 3:
                self.affine = reorient_affine(affine, self.shape, "LPS")
                self.affine[:2] *= -1
            self.affine[self.affine == 0] = 0.0
        except Exception:
            self.affine = None

    def get_resolution(self, head):
        return _get_resolution(head, self.fov, self.shape)


def get_plane_normal(orientation):  # noqa
    x, y = orientation
    normal = np.cross(x, y)
    normal = normal.round(7)
    normal[normal == 0] = 0.0
    return normal


def get_relative_slice_position(orientation, position):
    z = get_plane_normal(orientation)
    return z @ position


def get_fov(head):  # noqa
    encoding = head.encoding[-1]

    return [
        encoding.recon_space.field_of_view_mm.z,
        encoding.recon_space.field_of_view_mm.y,
        encoding.recon_space.field_of_view_mm.x,
    ]


def get_shape(head):  # noqa
    encoding = head.encoding[-1]

    return [
        encoding.recon_space.matrix_size.z,
        encoding.recon_space.matrix_size.y,
        encoding.recon_space.matrix_size.x,
    ]


def _get_resolution(head, fov, shape):  # noqa
    resolution = np.asarray(fov) / np.asarray(shape)
    if get_user_param(head, "SliceThickness"):
        resolution[0] = get_user_param(head, "SliceThickness")

    return resolution.tolist()


def get_image_position(head, raw_or_image_headers):
    axis_map = get_user_param(head, "AxisMaps")
    if "slice" in axis_map:
        slice_idx = axis_map["slice"]
    elif "kspace_encoding_step_2" in axis_map:
        slice_idx = axis_map["kspace_encoding_step_2"]
    else:
        slice_idx = None

    # get headers across slice axis
    if slice_idx is not None:
        _headers = raw_or_image_headers[..., None].swapaxes(slice_idx, -1)
        _headers = _headers.reshape(-1, _headers.shape[-1])
        _headers = _headers[0]
    else:
        _headers = [raw_or_image_headers.ravel()[0]]  # single slice case

    return np.stack([np.asarray(head.position) for head in _headers], axis=1)


def get_image_orientation(head, raw_or_image_headers):  # noqa
    axis_map = get_user_param(head, "AxisMaps")
    if "slice" in axis_map:
        slice_idx = axis_map["slice"]
    elif "kspace_encoding_step_2" in axis_map:
        slice_idx = axis_map["kspace_encoding_step_2"]
    else:
        slice_idx = None

    # get headers across slice axis
    if slice_idx is not None:
        _headers = raw_or_image_headers[..., None].swapaxes(slice_idx, -1)
        _headers = _headers.reshape(-1, _headers.shape[-1])
        _headers = _headers[0]
    else:
        _headers = [raw_or_image_headers.ravel()[0]]  # single slice case

    if "line_dir" in vars(raw_or_image_headers.ravel()[0]):
        return np.stack(
            [np.concatenate([head.line_dir, head.col_dir]) for head in _headers], axis=1
        )
    return np.stack(
        [np.concatenate([head.read_dir, head.phase_dir]) for head in _headers], axis=1
    )


def make_nifti_affine(shape, position, orientation, resolution):
    """
    Return affine transform between voxel coordinates and mm coordinates.

    Parameters
    ----------
    shape : tuple
        Volume shape (nz, ny, nx).
    resolution : tuple
        Image resolution in mm (dz, dy, dz).
    position : np.ndarray
        Position of each slice (3, nz).
    orientation : np.ndarray
        Image orientation matrix.

    Returns
    -------
    Affine matrix describing image position and orientation.

    References
    ----------
    [1] https://nipy.org/nibabel/dicom/spm_dicom.html#spm-volume-sorting
    """
    # get image size
    nz, ny, nx = shape

    # get resoluzion
    dz, dy, dx = resolution

    # common parameters
    T = position
    T1 = T[:, 0].round(4)

    F = orientation
    dr, dc = np.asarray([dy, dx]).round(4)

    if nz == 1:  # single slice case
        n = get_plane_normal(orientation)
        ds = float(dz)

        A0 = np.stack(
            (
                np.append(F[0] * dc, 0),
                np.append(F[1] * dr, 0),
                np.append(-ds * n, 0),
                np.append(T1, 1),
            ),
            axis=1,
        )

    else:  # multi slice case
        N = nz
        TN = T[:, -1].round(4)
        A0 = np.stack(
            (
                np.append(F[0] * dc, 0),
                np.append(F[1] * dr, 0),
                np.append((TN - T1) / (N - 1), 0),
                np.append(T1, 1),
            ),
            axis=1,
        )

    return A0.astype(np.float32)


def extract_rotation_from_affine(A):  # noqa
    R = A[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def extract_inplane_rotation(R):  # noqa
    return np.arctan2(R[1, 0], R[0, 0])


def reorient_affine(affine, shape, orientation):  # noqa
    tmp = np.ones(shape[-3:], dtype=np.float32)
    tmp = nib.Nifti1Image(tmp, affine)
    tmp = reorient_nifti(tmp, orientation)

    return tmp.affine


def reorient_nifti(input, orientation):
    # get input orientation
    orig_ornt = nib.io_orientation(input.affine)

    # get target orientation
    targ_ornt = axcodes2ornt(orientation)

    # estimate transform
    transform = ornt_transform(orig_ornt, targ_ornt)
    return input.as_reoriented(transform)


def detect_scan_orientation(image_orientation_patient):  # noqa
    row_cosines = np.array(
        image_orientation_patient[0]
    )  # Direction cosines of the rows
    col_cosines = np.array(
        image_orientation_patient[1]
    )  # Direction cosines of the columns

    # Calculate the slice normal (cross product of row and column direction cosines)
    slice_normal = np.cross(row_cosines, col_cosines)

    # Normalize the slice normal to unit vector
    slice_normal = slice_normal / np.linalg.norm(slice_normal)

    # Check if slice normal is close to one of the principal axes
    # Axial (slice normal should be along Z)
    if np.isclose(slice_normal[2], 1) or np.isclose(slice_normal[2], -1):
        orientation = "ax"
    # Sagittal (slice normal should be along X)
    elif np.isclose(slice_normal[0], 1) or np.isclose(slice_normal[0], -1):
        orientation = "sag"
    # Coronal (slice normal should be along Y)
    elif np.isclose(slice_normal[1], 1) or np.isclose(slice_normal[1], -1):
        orientation = "cor"
    else:
        # For oblique slices, compute the angle between the slice normal and each principal axis
        angles = {
            "ax": np.abs(slice_normal[2]),
            "sag": np.abs(slice_normal[0]),
            "cor": np.abs(slice_normal[1]),
        }
        # Find the orientation with the highest alignment (smallest angle)
        closest_orientation = max(angles, key=angles.get)
        orientation = closest_orientation

    return orientation
