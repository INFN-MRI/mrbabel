"""Siemens reading routines."""

__all__ = ["read_siemens"]

import glob
import warnings

import mrd

try:
    import twixtools

    __TWIXTOOLS_AVAILABLE__ = True
except Exception:
    __TWIXTOOLS_AVAILABLE__ = False


from ..._file_search import get_paths

from ..converters._siemens2mrd import read_siemens_header, read_siemens_acquisitions
from ..sorting import sort_kspace


def read_siemens(
    path: str,
    sort: bool = True,
    head_template: mrd.Header | None = None,
    acquisitions_template: list[mrd.Acquisition] | None = None,
    xml_file: str | None = None,
    xsl_file: str | None = None,
) -> tuple[mrd.ReconBuffer | list[mrd.ReconBuffer] | list[mrd.Acquisition], mrd.Header]:
    """
    Read input Siemens k-space file.

    Parameters
    ----------
    paths : str | list of str
        Path or list of file paths to Siemens files.
    sort : bool, optional
        If ``True``, sort list of MRD Acquisitions into a MRD ReconBuffer.
        The default is ``True``.
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
    if __TWIXTOOLS_AVAILABLE__ is False:
        print(
            "twixtools not found - install it with pip install git+https://github.com/pehses/twixtools.git"
        )
    if isinstance(path, str) and path.endswith(".dat"):
        path = glob.glob(path)
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
    twix_obj = twixtools.map_twix(path, verbose=False)
    head = read_siemens_header(twix_obj, head_template, xml_file, xsl_file)
    acquisitions = read_siemens_acquisitions(twix_obj, acquisitions_template)

    if sort:
        recon_buffers, head = sort_kspace(acquisitions, head)
        return recon_buffers, head

    return acquisitions, head
