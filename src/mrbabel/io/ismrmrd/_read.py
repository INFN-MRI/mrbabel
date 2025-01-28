"""ISMRMRD reading routines."""

__all__ = ["read_ismrmrd"]

import warnings

import ismrmrd
import ismrmrd.xsd

import mrd

from ..._file_search import get_paths

from ..converters._ismrmd2mrd import read_ismrmrd_header, read_ismrmrd_acquisitions
from ..sorting import sort_kspace


def read_ismrmrd(
    path: str, sort: bool = True
) -> mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition]:
    """
    Read input ISMRMRD k-space file.

    Parameters
    ----------
    paths : str | list of str
        Path or list of file paths to ISMRMRD files.
    sort : bool, optional
        If ``True``, sort list of MRD Acquisitions into a MRD ReconBuffer.
        The default is ``True``.

    Returns
    -------
    image : mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition]
        MRD ReconBuffer(s)  or list of MRD Acquisitions parsed from MRD file.
    head : mrd.Head
        MRD Header parsed from ISMRMRD files.

    """
    if isinstance(path, str) and path.endswith(".h5"):
        path = [path]
    else:
        path = get_paths("h5", path)
    if len(path) == 0:
        raise ValueError("ISMRMRD file not found in target directory.")
    if len(path) > 1:
        raise warnings.warn(
            f"Found multiple ({len(path)}) ISMRMRD files - picking the first",
            UserWarning,
        )
    path = path[0]

    # reading
    dset = ismrmrd.Dataset(path)
    head = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    acquisitions = [
        dset.read_acquisition(n) for n in range(dset.number_of_acquisitions())
    ]

    # convert to MRD (ISMRMRD v2)
    head = read_ismrmrd_header(head)
    acquisitions = read_ismrmrd_acquisitions(acquisitions)

    if sort:
        recon_buffers, head = sort_kspace(acquisitions, head)
        return recon_buffers, head

    return acquisitions, head
