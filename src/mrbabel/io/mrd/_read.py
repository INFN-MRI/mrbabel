"""MRD reading routines."""

__all__ = ["read_mrd"]

import warnings

import mrd

from ..._file_search import get_paths

from ..sorting import sort_kspace


def read_mrd(
    path: str, sort: bool = True
) -> mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition]:
    """
    Read input MRD k-space file.

    Parameters
    ----------
    paths : str | list of str
        Path or list of file paths to MRD files.
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
    if isinstance(path, str) and path.endswith(".bin"):
        path = [path]
    else:
        path = get_paths("bin", path)
    if len(path) == 0:
        raise ValueError("MRD file not found in target directory.")
    if len(path) > 1:
        raise warnings.warn(
            f"Found multiple ({len(path)}) MRD files - picking the first", UserWarning
        )
    path = path[0]

    # reading
    with mrd.BinaryMrdReader(path) as reader:
        head = reader.read_header()
        acq_stream = _acquisition_reader(reader.read_data())
        if head is None:
            raise Exception("Could not read header")
        acquisitions = [item for item in acq_stream]

    if sort:
        recon_buffers = sort_kspace(acquisitions, head)
        return recon_buffers, head

    return acquisitions, head


def _acquisition_reader(input: list[mrd.StreamItem]) -> list[mrd.Acquisition]:
    for item in input:
        if not isinstance(item, mrd.StreamItem.Acquisition):
            # Skip non-acquisition items
            continue
        if item.value.head.flags & mrd.AcquisitionFlags.IS_NOISE_MEASUREMENT:
            # Currently ignoring noise scans
            continue
        yield item.value
