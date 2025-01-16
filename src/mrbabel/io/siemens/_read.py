"""Siemens reading routines."""

__all__ = ["read_siemens"]

import warnings

import ismrmrd
import ismrmrd.xsd

import mrd
import mapvbvd

from ...data import sort_kspace
from ..._file_search import get_paths

from ..converters._siemens2mrd import read_siemens_header


def read_siemens(
    path: str, sort: bool = True
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

    Returns
    -------
    image : mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition]
        MRD ReconBuffer(s)  or list of MRD Acquisitions parsed from MRD file.
    head : mrd.Head
        MRD Header parsed from NIfTI files.

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
    twixObj = mapvbvd.mapVBVD(path, quiet=True)
    if isinstance(twixObj, list):
        twixObj = twixObj[-1]
    head = read_siemens_header(twixObj.hdr)

    return None, head    
    
    
    
    