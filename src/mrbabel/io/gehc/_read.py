"""GE HealthCare reading routines."""

__all__ = ["read_gehc"]

import warnings

import mrd

try:
    import getools
    __GEHC_AVAILABLE__ = True
except Exception:
    __GEHC_AVAILABLE__ = False

from ...data import sort_kspace
from ..._file_search import get_paths

from ..converters._gehc2mrd import read_gehc_header, read_gehc_acquisitions


def read_gehc(
    path: str,
    sort: bool = True,
) -> mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition]:
    """
    Read input GE HealthCare k-space file.

    Parameters
    ----------
    paths : str | list of str
        Path or list of file paths to GEHC files.
    sort : bool, optional
        If ``True``, sort list of MRD Acquisitions into a MRD ReconBuffer.
        The default is ``True``.

    Returns
    -------
    image : mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition]
        MRD ReconBuffer(s)  or list of MRD Acquisitions parsed from MRD file.
    head : mrd.Head
        MRD Header parsed from GEHC files.

    """
    if __GEHC_AVAILABLE__ is False:
        print("GEHC reader is private - ask for access at https://docs.google.com/forms/d/1BvA1h8qb9GmndqiXMplQbf3IujgBIehQ1psnfmW0tew/edit")
    if isinstance(path, str) and (path.endswith(".h5") or path.endswith(".7")):
        path = [path]
    else:
        path = get_paths("h5", path, "7")
    if len(path) == 0:
        raise ValueError("GEHC file not found in target directory.")
    if len(path) > 1:
        raise warnings.warn(
            f"Found multiple ({len(path)}) PFiles/ScanArchive files - picking the first", UserWarning
        )
    path = path[0]

    # reading
    gehc_raw, gehc_head = getools.read_rawdata(path)
    head = read_gehc_header(gehc_head)
    acquisitions = read_gehc_acquisitions(head, gehc_raw)

    if sort:
        recon_buffers = sort_kspace(acquisitions, head)
        return recon_buffers, head

    return acquisitions, head
