"""ISMRMRD reading routines."""

__all__ = ["read_ismrmrd"]

import glob
import warnings

try:
    import ismrmrd
    import ismrmrd.xsd

    __ISMRMRD_AVAILABLE__ = True
except Exception:
    __ISMRMRD_AVAILABLE__ = False


import mrd

from ..._file_search import get_paths

if __ISMRMRD_AVAILABLE__:
    from ..converters._ismrmd2mrd import read_ismrmrd_header, read_ismrmrd_acquisitions
from ..sorting import sort_kspace


def read_ismrmrd(
    path: str,
    sort: bool = True,
    head_template: mrd.Header | None = None,
    acquisitions_template: list[mrd.Acquisition] | None = None,
) -> tuple[mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition], mrd.Header]:
    """
    Read input ISMRMRD k-space file.

    Parameters
    ----------
    paths : str | list of str
        Path or list of file paths to ISMRMRD files.
    head_template : mrd.Header | None, optional
        MRD Header as defined at sequence design step. If provided,
        uses it as a blueprint to define encoding limits and sequence parameters.
        It is update with scan specific info (SubjectInformation, etc) from raw data.
        The default is ``None`` (uses raw header only).
    acquisitions_template : list[mrd.Acquisition] | None, optional
        MRD Acquisition(s) as defined at sequence design step. If provided,
        uses it as a blueprint to define data ordering.
        It is update with scan specific info (orientation, etc) from raw data.
        The default is ``None`` (uses raw header only).
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
    if __ISMRMRD_AVAILABLE__ is False:
        raise ValueError("ismrmrd not found - install it with 'pip install ismrmrd'")
    if isinstance(path, str) and path.endswith(".h5"):
        path = glob.glob(path)
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
    head = read_ismrmrd_header(head, head_template)
    acquisitions = read_ismrmrd_acquisitions(acquisitions, acquisitions_template)

    if sort:
        recon_buffers, head = sort_kspace(acquisitions, head)
        return recon_buffers, head

    return acquisitions, head
