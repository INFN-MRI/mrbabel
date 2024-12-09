"""Test MRD Images parsing from DICOM."""

import pytest
import pydicom

import numpy as np

from mrbabel.io.dicom._read import read_dicom_images
from mrbabel._file_search import get_paths
from mrbabel import testdata as _testdata

import mrd


@pytest.fixture
def load_real_dicom_files():
    """Fixture to load real DICOM files."""
    test_dir = _testdata(name="dicom")
    dicom_files = get_paths("dcm", test_dir)
    return [pydicom.dcmread(f) for f in dicom_files]


@pytest.fixture
def create_mrd_header():
    """Create a basic MRD header for testing."""
    header = mrd.Header()
    header.sequence_parameters = mrd.SequenceParametersType()
    header.encoding = [mrd.EncodingType()]
    header.encoding[0].encoded_space = mrd.EncodingSpaceType()
    header.encoding[0].recon_space = mrd.EncodingSpaceType()
    header.encoding[0].encoding_limits = mrd.EncodingLimitsType()
    header.encoding[0].encoding_limits.slice = mrd.LimitType()
    header.encoding[0].encoding_limits.contrast = mrd.LimitType()
    header.encoding[0].encoding_limits.kspace_encoding_step_0 = mrd.LimitType()
    header.encoding[0].encoding_limits.kspace_encoding_step_1 = mrd.LimitType()

    return header


def test_read_dicom_images_basic(load_real_dicom_files, create_mrd_header):
    """Test that images are correctly converted from DICOM to MRD format."""
    dicom_datasets = load_real_dicom_files
    mrd_header = create_mrd_header

    # Call the function
    images, updated_header = read_dicom_images(dicom_datasets, mrd_header)

    # Verify the number of images matches the number of DICOM datasets
    assert len(images) == len(
        dicom_datasets
    ), "Number of images does not match the number of DICOM datasets."


def test_instance_number_ordering(load_real_dicom_files, create_mrd_header):
    """Test that images are correctly ordered by InstanceNumber."""
    dicom_datasets = load_real_dicom_files
    mrd_header = create_mrd_header

    # Call the function
    images, _ = read_dicom_images(dicom_datasets, mrd_header)

    # Check that the images are sorted by InstanceNumber
    instance_numbers = [int(image.head.image_index) for image in images]
    assert instance_numbers == sorted(
        instance_numbers
    ), "Images are not ordered by InstanceNumber."


def test_contrast_parameters(load_real_dicom_files, create_mrd_header):
    """Test that unique contrast parameters are correctly extracted."""
    dicom_datasets = load_real_dicom_files
    mrd_header = create_mrd_header

    # Call the function
    _, updated_header = read_dicom_images(dicom_datasets, mrd_header)

    # Verify contrast parameters
    flip_angles = np.unique([float(dset.FlipAngle) for dset in dicom_datasets])
    echo_times = np.unique([float(dset.EchoTime) for dset in dicom_datasets])
    repetition_times = np.unique(
        [float(dset.RepetitionTime) for dset in dicom_datasets]
    )

    assert np.allclose(
        updated_header.sequence_parameters.flip_angle_deg, list(flip_angles)
    ), "Flip angles do not match."
    assert np.allclose(
        updated_header.sequence_parameters.t_e, list(echo_times)
    ), "Echo times do not match."
    assert np.allclose(
        updated_header.sequence_parameters.t_r, list(repetition_times)
    ), "Repetition times do not match."


def test_slice_indexing(load_real_dicom_files, create_mrd_header):
    """Test that slice indexing is correct."""
    dicom_datasets = load_real_dicom_files
    mrd_header = create_mrd_header

    # Call the function
    images, _ = read_dicom_images(dicom_datasets, mrd_header)

    # Verify slice indexing
    slice_indices = [image.head.slice for image in images]
    slice_locations = [dset.SliceLocation for dset in dicom_datasets]
    assert len(slice_indices) == len(slice_locations), "Slice indexing mismatch."


def test_image_type_extraction(load_real_dicom_files, create_mrd_header):
    """Test that the correct image type is extracted for each image."""
    dicom_datasets = load_real_dicom_files
    mrd_header = create_mrd_header

    # Call the function
    images, _ = read_dicom_images(dicom_datasets, mrd_header)

    # Verify image types
    for image in images:
        assert image.head.image_type in [
            mrd.ImageType.MAGNITUDE,
            mrd.ImageType.PHASE,
        ], f"Invalid image type: {image.head.image_type}"


def test_field_of_view(load_real_dicom_files, create_mrd_header):
    """Test that the field of view (FOV) is calculated correctly."""
    dicom_datasets = load_real_dicom_files
    mrd_header = create_mrd_header

    # Call the function
    images, _ = read_dicom_images(dicom_datasets, mrd_header)

    # Verify field of view
    for i, dset in enumerate(dicom_datasets):
        expected_fov = (
            dset.PixelSpacing[0] * dset.Rows,
            dset.PixelSpacing[1] * dset.Columns,
            dset.SliceThickness,
        )
        assert np.allclose(
            images[i].head.field_of_view, expected_fov
        ), f"FOV mismatch for image {i}"
