"""DICOM to NIfTI conversion utilities."""

__all__ = ["get_json_template"]

import json

from os.path import dirname
from os.path import join as pjoin

import mrd


def get_json_template(head: mrd.Header) -> dict:
    """
    Get JSON sidecar template according to system vendor.

    Parameters
    ----------
    head : mrd.Header
        Input MRD header to determine manufacturer.

    Returns
    -------
    dict
        JSON sidecar template dictionary.

    """
    if head.acquisition_system_information.system_vendor:
        if "GE" in head.acquisition_system_information.system_vendor.upper():
            json_file = pjoin(dirname(__file__), "_nifti_json", "GEHC.json")
        if "SIEMENS" in head.acquisition_system_information.system_vendor.upper():
            json_file = pjoin(dirname(__file__), "_nifti_json", "SIEMENS.json")
        if "PHILIPS" in head.acquisition_system_information.system_vendor.upper():
            json_file = pjoin(dirname(__file__), "_nifti_json", "PHILIPS.json")
        else:
            json_file = pjoin(dirname(__file__), "_nifti_json", "common.json")
    else:
        json_file = pjoin(dirname(__file__), "_nifti_json", "common.json")

    with open(json_file, "r") as f:
        json_template = json.load(f)

    return json_template
