"""Data selection subroutines."""

__all__ = ["select_readout_pts"]

import mrd


def select_readout_pts(recon_buffer: mrd.ReconBuffer) -> mrd.ReconBuffer:
    """
    Select readout points according to Acquisition Header.

    Assumes that all data in the buffer have the same ADC.

    Parameters
    ----------
    recon_buffer : mrd.ReconBuffer
        Input ReconBuffer.

    Returns
    -------
    mrd.ReconBuffer
        Trimmed ReconBuffer.

    """
    discard_pre = recon_buffer.headers.ravel()[0].discard_pre
    discard_post = recon_buffer.headers.ravel()[0].discard_post
    if discard_post == 0:
        recon_buffer.data = recon_buffer.data[..., discard_pre:]
    else:
        recon_buffer.data = recon_buffer.data[..., discard_pre:-discard_post]

    return recon_buffer
