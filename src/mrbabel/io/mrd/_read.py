"""MRD reading routines."""

__all__ = ["read_mrd"]

import glob
import warnings

import mrd

from ..._file_search import get_paths

from ..sorting import sort_kspace


def read_mrd(
    path: str,
    sort: bool = True,
    head_template: mrd.Header | None = None,
    acquisitions_template: list[mrd.Acquisition] | None = None,
) -> tuple[mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition], mrd.Header]:
    """
    Read input MRD k-space file.

    Parameters
    ----------
    paths : str | list of str
        Path or list of file paths to MRD files.
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
        MRD Header parsed from NIfTI files.

    """
    if isinstance(path, str) and path.endswith(".bin"):
        path = glob.glob(path)
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

    # update header
    if head_template is not None:
        # replace encoding
        head.encoding = head_template.encoding

        # replace contrast
        head.sequence_parameters = head_template.sequence_parameters

        # update user parameters
        head.user_parameters.extend(head_template.user_parameters)

    # update acquisitions
    if acquisitions_template is not None:
        nacquisitions = len(acquisitions)
        for n in range(nacquisitions):
            acquisitions[n].head.flags = acquisitions_template[n].head.flags
            acquisitions[n].head.idx.kspace_encode_step_1 = acquisitions_template[
                n
            ].head.idx.kspace_encode_step_1
            acquisitions[n].head.idx.kspace_encode_step_2 = acquisitions_template[
                n
            ].head.idx.kspace_encode_step_2
            acquisitions[n].head.idx.slice = acquisitions_template[n].head.idx.slice
            acquisitions[n].head.idx.contrast = acquisitions_template[
                n
            ].head.idx.contrast
            acquisitions[n].head.discard_pre = acquisitions_template[n].head.discard_pre
            acquisitions[n].head.discard_post = acquisitions_template[
                n
            ].head.discard_post
            acquisitions[n].head.center_sample = acquisitions_template[
                n
            ].head.center_sample
            acquisitions[n].head.encoding_space_ref = acquisitions_template[
                n
            ].head.encoding_space_ref
            acquisitions[n].head.sample_time_us = acquisitions_template[
                n
            ].head.sample_time_us

    if sort:
        recon_buffers, head = sort_kspace(acquisitions, head)
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
