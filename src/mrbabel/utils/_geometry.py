"""Geometry estimation tools.

Adapted from https://github.com/pehses/twixtools/blob/master/twixtools/geometry.py
"""

__all__ = ["get_geometry"]

import numpy as np
import nibabel as nib

from nibabel.orientations import axcodes2ornt, ornt_transform

import mrd

from ._user import get_user_param

pcs_directions = ["dSag", "dCor", "dTra"]

# p. 418 - pcs to dcs
pcs_transformations = {
    "HFS": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    "HFP": [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    "FFS": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
}


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
    """
    Get geometric information from MRD Header.

    During initialization, information about slice geometry is copied from the supplied twix dict.
    Methods for conversion between the different coordinate systems
    Patient Coordinate System (PCS; Sag/Cor/Tra), Device Coordinate System (XYZ) and Gradient Coordinate System
    (GCS or PRS; Phase, Readout, Slice) are implemented (so far only rotation, i.e. won't work for offcenter measurementes).

    Examples
    --------
    ```
    import twixtools
    twix = twixtools.read_twix('meas.dat', parse_geometry=True, parse_data=False)
    x = [1,1,1]
    y = twix[-1]['geometry'].rps_to_xyz() @ x
    ```

    Based on work from Christian Mirkes and Ali Aghaeifar.
    """

    def __init__(self, head, raw_or_image_headers):  # noqa
        self.from_mrd(head, raw_or_image_headers)

    def __repr__(self):  # noqa
        return (
            "Geometry:\n"
            f"  inplane_rot: {self.inplane_rot}\n"
            f"  normal: {self.normal}\n"
            f"  offset: {self.offset}\n"
            f"  patient_position: {self.patient_position}\n"
            f"  rotmatrix: {self.rotmatrix}\n"
            f"  voxelsize: {self.voxelsize}"
        )

    def __str__(self):  # noqa
        return (
            "Geometry:\n"
            f"  inplane_rot: {self.inplane_rot}\n"
            f"  normal: {self.normal}\n"
            f"  offset: {self.offset}\n"
            f"  patient_position: {self.patient_position}\n"
            f"  rotmatrix: {self.rotmatrix}\n"
            f"  voxelsize: {self.voxelsize}"
        )

    def from_mrd(self, head, raw_or_image_headers):  # noqa
        if get_user_param(head, "mode"):
            self.dims = int(get_user_param(head, "mode")[0])
        else:
            self.dims = None
        self.fov = get_fov(head)
        self.shape = get_shape(head)
        self.voxelsize = self.get_resolution(head)

        # get normal
        orientation = get_image_orientation(raw_or_image_headers)
        self.normal = get_plane_normal(orientation).tolist()

        # get position
        positions = get_position(head, raw_or_image_headers)
        self.offset = positions.mean(axis=-1).tolist()

        # get patient position
        self.patient_position = None
        if head.measurement_information:
            if head.measurement_information.patient_position.name is not None:
                self.patient_position = "".join(
                    head.measurement_information.patient_position.name.split("_")
                )

        affine = make_nifti_affine(self.shape, positions, orientation, self.voxelsize)
        self.rotmatrix = extract_rotation_from_affine(affine)
        self.inplane_rot = extract_inplane_rotation(self.rotmatrix)
        self.shape = self.shape[::-1]
        self.voxelsize = self.voxelsize[::-1]
        self.fov = self.fov[::-1]

        # reorient affine
        self.affine = affine
        self.affine[self.affine == 0] = 0.0
        self.affine = reorient_affine(self.shape, affine, "RAS")

    def get_resolution(self, head):
        return _get_resolution(head, self.fov, self.shape)


def get_plane_normal(orientation):  # noqa
    x, y = orientation
    normal = np.cross(x, y)
    normal[normal == 0] = np.abs(normal[normal == 0])
    return normal


def get_relative_slice_position(orientation, position):
    z = get_plane_normal(orientation)
    return z @ position


def get_position(head, raw_or_image_headers):
    axis_map = get_user_param(head, "AxisMaps")
    if "slice" in axis_map:
        slice_idx = axis_map["slice"]
    elif "kspace_encoding_step_2" in axis_map["slice"]:
        slice_idx = axis_map["kspace_encoding_step_2"]
    else:
        slice_idx = None

    # get headers across slice axis
    if slice_idx:
        _headers = raw_or_image_headers[..., None].swapaxes(slice_idx, -1)
        _headers = _headers.reshape(-1, _headers.shape[-1])
        _headers = _headers[0]
    else:
        _headers = raw_or_image_headers.ravel()[0]  # single slice case

    return np.stack([np.asarray(head.position) for head in _headers], axis=1)


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


def get_image_orientation(raw_or_image_headers, astuple=False):  # noqa
    _header = raw_or_image_headers.ravel()[0]
    F = np.concatenate([_header.read_dir, _header.phase_dir]).reshape(2, 3)

    if astuple:
        F = tuple(F.ravel())

    return np.around(F, 4)


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

    # sign of affine matrix
    # A0[:2, :] *= -1

    return A0.astype(np.float32)


def extract_rotation_from_affine(A):  # noqa
    R = A[:3, :3]
    # R[:2, :] *= -1
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def extract_inplane_rotation(R):  # noqa
    return np.arctan2(R[1, 0], R[0, 0])


def reorient_affine(shape, affine, orientation):  # noqa
    # get input orientation
    orig_ornt = nib.io_orientation(affine)

    # get target orientation
    targ_ornt = axcodes2ornt(orientation)

    # estimate transform
    transform = ornt_transform(orig_ornt, targ_ornt)

    # reorient
    tmp = np.ones(shape[-3:], dtype=np.float32)
    tmp = nib.Nifti1Image(tmp, affine)
    tmp = tmp.as_reoriented(transform)

    return tmp.affine
