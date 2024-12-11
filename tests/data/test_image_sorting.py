"""Test image sorting."""

import pytest
import numpy as np
from types import SimpleNamespace

import mrd

from mrbabel.data.operations import sort_images


# Fixtures for creating test data
@pytest.fixture
def create_mock_images():
    """
    Create mock MRD.Image objects for testing.
    Returns a list of mock MRD.Image objects with varying image types, slice indices,
    and contrast indices.
    """

    def _create_mock_images(n_slices, n_contrasts, image_type):
        images = []
        for contrast in range(n_contrasts):
            for slice_idx in range(n_slices):
                # Mock header
                head = mrd.ImageHeader(
                    image_type=image_type, slice=slice_idx, contrast=contrast
                )
                # Mock data
                data = np.random.rand(128, 128).astype(np.float32)
                # Mock metadata
                meta = {"contrast": contrast, "slice": slice_idx}

                images.append(mrd.Image(head=head, data=data, meta=meta))
        return images

    return _create_mock_images


@pytest.fixture
def head():
    return SimpleNamespace(acquisition_system_information=None)


# Test: Basic sorting functionality
def test_sort_images_basic(create_mock_images, head):
    n_slices = 5
    n_contrasts = 3
    images = create_mock_images(n_slices, n_contrasts, mrd.ImageType.MAGNITUDE)

    result = sort_images(images, head)

    # Validate dimensions
    assert result.data.shape == (
        n_contrasts,
        n_slices,
        128,
        128,
    ), "Incorrect data shape."

    # Validate headers and metadata consistency
    assert len(result.headers) == len(images), "Header mismatch."
    assert len(result.meta) == len(images), "Metadata mismatch."

    # Validate data integrity
    for contrast in range(n_contrasts):
        for slice_idx in range(n_slices):
            expected_data = images[contrast * n_slices + slice_idx].data
            actual_data = result.data[contrast, slice_idx, :, :]
            np.testing.assert_array_equal(expected_data, actual_data, "Data mismatch.")


# Test: Complex-valued images
def test_sort_images_complex(create_mock_images, head):
    n_slices = 4
    n_contrasts = 2
    images = create_mock_images(n_slices, n_contrasts, mrd.ImageType.COMPLEX)

    result = sort_images(images, head)

    # Validate dimensions
    assert result.data.shape == (
        n_contrasts,
        n_slices,
        128,
        128,
    ), "Incorrect data shape for complex images."

    # Validate data type
    assert result.data.dtype == np.complex64, "Data type mismatch for complex images."


# Test: Mixed real/imag images
def test_sort_images_mixed_real_imag(create_mock_images, head):
    n_slices = 3
    n_contrasts = 2
    real_images = create_mock_images(n_slices, n_contrasts, mrd.ImageType.REAL)
    imag_images = create_mock_images(n_slices, n_contrasts, mrd.ImageType.IMAG)
    images = real_images + imag_images

    result = sort_images(images, head)

    # Validate dimensions
    assert result.data.shape == (
        n_contrasts,
        n_slices,
        128,
        128,
    ), "Incorrect data shape for mixed real/imag."

    # Validate data type
    assert result.data.dtype == np.complex64, "Data type mismatch for mixed real/imag."


# Test: Invalid input handling
def test_sort_images_invalid_input(create_mock_images, head):
    with pytest.raises(RuntimeError, match="Mixing real and complex-valued images"):
        # Create mixed real and complex images, which should raise an error
        real_images = create_mock_images(3, 2, mrd.ImageType.REAL)
        complex_images = create_mock_images(3, 2, mrd.ImageType.COMPLEX)
        images = real_images + complex_images
        sort_images(images, head)
