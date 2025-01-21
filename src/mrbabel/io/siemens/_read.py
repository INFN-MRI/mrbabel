"""Siemens reading routines."""

__all__ = ["read_siemens"]

import warnings

import ismrmrd
import ismrmrd.xsd

import mrd
import twixtools

from ...data import sort_kspace
from ..._file_search import get_paths

from ..converters._siemens2mrd import read_siemens_header, read_siemens_acquisitions


def read_siemens(
    path: str,
    sort: bool = True,
    xml_file: str | None = None,
    xsl_file: str | None = None,
) -> mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition]:
    """
    Read input Siemens k-space file.

    Parameters
    ----------
    paths : str | list of str
        Path or list of file paths to Siemens files.
    sort : bool, optional
        If ``True``, sort list of MRD Acquisitions into a MRD ReconBuffer.
        The default is ``True``.
    xml_file : str | None, optional
        XML file to create xml string from Twix header. If not
        provided, uses the same default as "siemens_to_ismrmrd".
        The default is ``None``.
    xsl_file : str | None, optional
        XSLT file to convert Twix xml string to MRD. If not
        provided, uses the same default as "siemens_to_ismrmrd".
        The default is ``None``.

    Returns
    -------
    image : mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition]
        MRD ReconBuffer(s)  or list of MRD Acquisitions parsed from MRD file.
    head : mrd.Head
        MRD Header parsed from Siemens files.

    """
    if isinstance(path, str) and path.endswith(".dat"):
        path = [path]
    else:
        path = get_paths("dat", path)
    if len(path) == 0:
        raise ValueError("Siemens file not found in target directory.")
    if len(path) > 1:
        raise warnings.warn(
            f"Found multiple ({len(path)}) dat files - picking the first", UserWarning
        )
    path = path[0]

    # reading
    twix_obj = twixtools.read_twix(path)
    if isinstance(twix_obj, list):
        twix_obj = twix_obj[-1]
    head = read_siemens_header(twix_obj["hdr"])
    acquisitions = read_siemens_acquisitions(twix_obj["hdr"], twix_obj["mdb"])

    if sort:
        recon_buffers = sort_kspace(acquisitions, head)
        return recon_buffers, head

    return acquisitions, head
