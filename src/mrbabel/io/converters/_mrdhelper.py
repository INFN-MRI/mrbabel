"""
"""

import mrd


def raw2image(acq_head, image_index):
    """Populate ImageHeader fields from AcquisitionHeader."""
    img_head = mrd.ImageHeader()

    img_head.version = acq_head.version
    img_head.flags = acq_head.flags
    img_head.measurement_uid = acq_head.measurement_uid

    img_head.position = acq_head.position
    img_head.read_dir = acq_head.read_dir
    img_head.phase_dir = acq_head.phase_dir
    img_head.slice_dir = acq_head.slice_dir
    img_head.patient_table_position = acq_head.patient_table_position

    img_head.average = acq_head.idx.average
    img_head.slice = acq_head.idx.slice
    img_head.contrast = acq_head.idx.contrast
    img_head.phase = acq_head.idx.phase
    img_head.repetition = acq_head.idx.repetition
    img_head.set = acq_head.idx.set

    img_head.acquisition_time_stamp = acq_head.acquisition_time_stamp
    img_head.physiology_time_stamp = acq_head.physiology_time_stamp

    # Defaults, to be updated by the user
    img_head.image_type = mrd.ImageType.COMPLEX
    img_head.image_index = image_index
    img_head.image_series_index = 0

    img_head.user_float = acq_head.user_float
    img_head.user_int = acq_head.user_int

    return img_head


def image2raw(img_head):
    """Reconstruct AcquisitionHeader fields from ImageHeader."""
    acq_head = mrd.AcquisitionHeader()

    acq_head.version = img_head.version
    acq_head.flags = img_head.flags
    acq_head.measurement_uid = img_head.measurement_uid

    acq_head.position = img_head.position
    acq_head.read_dir = img_head.read_dir
    acq_head.phase_dir = img_head.phase_dir
    acq_head.slice_dir = img_head.slice_dir
    acq_head.patient_table_position = img_head.patient_table_position

    acq_head.idx.average = img_head.average
    acq_head.idx.slice = img_head.slice
    acq_head.idx.contrast = img_head.contrast
    acq_head.idx.phase = img_head.phase
    acq_head.idx.repetition = img_head.repetition
    acq_head.idx.set = img_head.set

    acq_head.acquisition_time_stamp = img_head.acquisition_time_stamp
    acq_head.physiology_time_stamp = img_head.physiology_time_stamp

    # Restore user-defined fields
    acq_head.user_float = img_head.user_float
    acq_head.user_int = img_head.user_int

    return acq_head
