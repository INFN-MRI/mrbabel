"""NIfTI to MRD Conversion Utilities."""

__all__ = [
    "read_nifti_header",
    "read_nifti_image",
]

import base64
import copy
import json as jsonlib
import warnings

import numpy as np
import mrd

from ._dicom2mrd import (
    read_dicom_header,
    read_dicom_images,
    _get_unique_contrasts,
    IMTYPE_MAPS,
)


def read_nifti_header(nii_data: np.ndarray, nii_head: dict) -> mrd.Header:
    """Create MRD Header from a NIfTI file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        head = read_dicom_header(nii_head["dset"])

    # Get sequence parameters from json
    flip_angle_deg = [float(json["FlipAngle"]) for json in nii_head["json"]]
    t_r = [float(json["RepetitionTime"]) * 1000.0 for json in nii_head["json"]]
    t_e = [float(json["EchoTime"]) * 1000.0 for json in nii_head["json"]]
    if "InversionTime" in nii_head["json"][0]:
        t_i = [float(json["InversionTime"]) * 1000.0 for json in nii_head["json"]]
    else:
        t_i = []

    # Get unique contrasts
    if len(t_i) > 0:
        contrasts = np.stack([flip_angle_deg, t_r, t_e, t_i], axis=1)
        u_contrasts, contrast_idx = _get_unique_contrasts(contrasts)
        flip_angle_deg, t_r, t_e, t_i = u_contrasts
    else:
        contrasts = np.stack([flip_angle_deg, t_r, t_e], axis=1)
        u_contrasts, contrast_idx = _get_unique_contrasts(contrasts)
        flip_angle_deg, t_r, t_e = u_contrasts

    # Remove duplicates
    n_contrasts = len(flip_angle_deg)
    flip_angle_deg = np.unique(flip_angle_deg).tolist()
    t_e = np.unique(t_e).tolist()
    t_r = np.unique(t_r).tolist()
    if len(t_i) > 0:
        t_i = np.unique(t_i).tolist()

    # Update sequence parameters
    if head.sequence_parameters is None:
        head.sequence_parameters = mrd.SequenceParametersType()
    head.sequence_parameters.flip_angle_deg = flip_angle_deg
    head.sequence_parameters.t_r = t_r
    head.sequence_parameters.t_e = t_e
    head.sequence_parameters.t_i = t_i

    # Update Z fov and matrix size
    n_slices = len(nii_head["ImagePositionPatient"])
    head.encoding[-1].encoded_space.field_of_view_mm.z *= n_slices
    head.encoding[-1].encoded_space.matrix_size.z *= n_slices

    head.encoding[-1].recon_space.field_of_view_mm.z *= n_slices
    head.encoding[-1].recon_space.matrix_size.z *= n_slices

    # Update encoding limits
    head.encoding[-1].encoding_limits.slice.maximum = n_slices - 1
    head.encoding[-1].encoding_limits.slice.center = n_slices // 2

    head.encoding[-1].encoding_limits.contrast.maximum = n_contrasts - 1
    head.encoding[-1].encoding_limits.contrast.center = n_contrasts // 2

    # Get vendor
    vendor = nii_head["json"][0]["Manufacturer"]

    def get_image_type(item):
        if "GE" in vendor.upper():
            try:
                return IMTYPE_MAPS["default"][item[3][0]]
            except Exception:
                return mrd.ImageType.MAGNITUDE
        return IMTYPE_MAPS["default"][item[2][0]]

    image_types = [get_image_type(json["ImageType"]).value for json in nii_head["json"]]
    image_types = np.asarray(image_types)

    # Sort images and jsons
    n_volumes = nii_data.shape[0]

    # Real-valued image
    is_complex = False
    if sum(image_types == 1) == n_volumes or sum(image_types == 3) == n_volumes:
        nii_data = nii_data.astype(np.complex64)

    # Complex images with separate magn/phase or real/imag
    else:
        is_complex = True
        _nii_data = [None, None, None, None]
        _json = []
        _contrast_idx = []
        if sum(image_types == 1) > 0:
            _nii_data[0] = np.zeros(
                (sum(image_types == 1), *nii_data[0].shape), dtype=np.float32
            )
        if sum(image_types == 2) > 0:
            _nii_data[1] = np.zeros(
                (sum(image_types == 2), *nii_data[0].shape), dtype=np.float32
            )
        if sum(image_types == 3) > 0:
            _nii_data[2] = np.zeros(
                (sum(image_types == 3), *nii_data[0].shape), dtype=np.float32
            )
        if sum(image_types == 4) > 0:
            _nii_data[3] = np.zeros(
                (sum(image_types == 4), *nii_data[0].shape), dtype=np.float32
            )

        # Sorting
        for n in range(n_volumes):
            idx = image_types[n] - 1
            cidx = contrast_idx[n]
            _nii_data[idx][cidx] = nii_data[n].astype(np.float32)
            if idx == 0:  # magnitude
                _json.append(nii_head["json"][n])
                _contrast_idx.append(cidx)

        # Real + 1j * Imag
        if sum(image_types == 3) > 0 and sum(image_types == 4) > 0:
            nii_data = _nii_data[2] + 1j * _nii_data[3]
        else:
            mag = _nii_data[0].view()
            phase = _nii_data[1].view()
            phase = (phase - phase.min()) / (phase.max() - phase.min())
            phase = 2 * np.pi * phase - np.pi
            nii_data = mag * np.exp(1j * phase)

    # Sort contrasts
    _order = np.argsort(_contrast_idx)
    nii_data = nii_data[_order]
    _json = np.asarray(_json)[_order].tolist()

    # Correct phase shift along z for GE systems
    if (
        is_complex
        and head.acquisition_system_information
        and head.acquisition_system_information.system_vendor
    ):
        if "GE" in head.acquisition_system_information.system_vendor.upper():
            mag = np.abs(nii_data.view())
            phase = np.angle(nii_data.view())
            phase[..., 1::2, :, :] = (
                (1e5 * (phase[..., 1::2, :, :] + 2 * np.pi)) % (2 * np.pi * 1e5)
            ) / 1e5 - np.pi
            nii_data = mag * np.exp(1j * phase)

    # Enforce single precision
    nii_data = nii_data.astype(np.complex64)

    # Assign json
    nii_head["json"] = _json

    return nii_data, nii_head, head


def read_nifti_image(
    nii_data: np.ndarray, nii_head: dict, head: mrd.Header
) -> mrd.ImageArray:
    """Create MRD ImageArray from a NIfTI file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image_template, _ = read_dicom_images([nii_head["dset"]], copy.deepcopy(head))

    # Get image_header and image_meta templates
    image_head_template = image_template[0].head
    image_meta_template = image_template[0].meta
    image_meta_template.pop("DicomJson", None)

    # Get image position patient
    ipp = nii_head["ImagePositionPatient"]

    # Get sizes
    n_contrasts = nii_data.shape[0]
    if len(nii_data.shape) == 5:
        n_phases = nii_data.shape[1]
        n_slices = nii_data.shape[2]
    else:
        n_phases = 1
        n_slices = nii_data.shape[1]

    # Construct headers
    image_head = np.empty((n_contrasts, n_phases, n_slices), dtype=object)
    image_meta = np.empty((n_contrasts, n_phases, n_slices), dtype=object)

    # Construction
    for contrast_idx in range(n_contrasts):
        for phase_idx in range(n_phases):
            for slice_idx in range(n_slices):
                # image head
                _image_head = copy.deepcopy(image_head_template)
                _image_head.image_type = mrd.ImageType.COMPLEX
                _image_head.position = ipp[slice_idx]
                _image_head.contrast = contrast_idx
                _image_head.phase = phase_idx
                _image_head.slice = slice_idx

                # Flags
                defs = mrd.ImageFlags
                if contrast_idx == 0:
                    _image_head.flags = defs.FIRST_IN_CONTRAST
                if contrast_idx == n_contrasts - 1:
                    _image_head.flags = defs.LAST_IN_CONTRAST
                if phase_idx == 0:
                    _image_head.flags = defs.FIRST_IN_PHASE
                if phase_idx == n_phases - 1:
                    _image_head.flags = defs.LAST_IN_PHASE
                if slice_idx == 0:
                    _image_head.flags = defs.FIRST_IN_SLICE
                if slice_idx == n_slices - 1:
                    _image_head.flags = defs.LAST_IN_SLICE

                image_head[contrast_idx, phase_idx, slice_idx] = _image_head

                # image meta
                _image_meta = copy.deepcopy(image_meta_template)
                image_meta[contrast_idx, phase_idx, slice_idx] = _image_meta

    # Transpose
    if len(nii_data.shape) == 5:
        nii_data = nii_data.swapaxes(0, 1)
    else:
        nii_data = nii_data[None, ...]
    nii_data = np.flip(nii_data, (-2, -1))
    nii_data = np.ascontiguousarray(nii_data)

    # assign
    image_data = nii_data.view()
    image_head = image_head.swapaxes(0, 1)
    image_meta = image_meta.swapaxes(0, 1)

    # Add axis map
    axis_map_keys = ["phase", "contrast", "slice", "rows", "columns"]
    axis_map_keys = np.asarray(axis_map_keys)
    singleton_axis = np.asarray(image_data.shape) == 1
    axis_map_keys = axis_map_keys[np.logical_not(singleton_axis)].tolist()
    axis_map_values = np.arange(len(axis_map_keys)).tolist()
    axis_map = dict(zip(axis_map_keys, axis_map_values))
    axis_map = base64.b64encode(jsonlib.dumps(axis_map).encode("utf-8")).decode("utf-8")
    if head.user_parameters is None:
        head.user_parameters = mrd.UserParametersType()
    head.user_parameters.user_parameter_base64.append(
        mrd.UserParameterBase64Type(name="AxisMaps", value=axis_map)
    )

    return (
        mrd.ImageArray(
            data=image_data.squeeze(),
            headers=image_head.squeeze(),
            meta=image_meta.squeeze(),
        ),
        head,
    )
