"""Test image unsorting."""

import pytest
import numpy as np

import mrd

from mrbabel.io.sorting import unsort_images


# Mock for MRD classes
class MockImageHeader:
    def __init__(self, image_type, contrast, slice):
        self.image_type = image_type
        self.contrast = contrast
        self.slice = slice


class MockImageArray:
    def __init__(self, data, headers, meta):
        self.data = data
        self.headers = headers
        self.meta = meta


class MockImage:
    def __init__(self, data, head, meta):
        self.data = data
        self.head = head
        self.meta = meta


@pytest.fixture
def mock_image_array():
    """Create a mock MRD ImageArray."""
    mock_data = np.random.rand(3, 5, 128, 128).astype(
        np.complex64
    )  # 3 contrasts, 5 slices, 128x128
    mock_headers = [
        MockImageHeader(image_type=mrd.ImageType.MAGNITUDE, contrast=i % 3, slice=i % 5)
        for i in range(15)
    ]
    mock_meta = [{"meta_key": f"meta_value_{i}"} for i in range(15)]
    return MockImageArray(data=mock_data, headers=mock_headers, meta=mock_meta)


# Test: Unsorting Images
def test_unsort_images(mock_image_array):
    result = unsort_images(mock_image_array)
    assert isinstance(result, list), "Output should be a list."
    assert len(result) == len(
        mock_image_array.headers
    ), "Number of images should match the number of headers."

    for i, img in enumerate(result):
        assert isinstance(
            img, mrd.Image
        ), "Each item in the output list should be an MRD Image."
        assert (
            img.head == mock_image_array.headers[i]
        ), "Header mismatch in the output images."
        assert (
            img.meta == mock_image_array.meta[i]
        ), "Meta information mismatch in the output images."
        assert img.data.shape == (128, 128), "Output image data should be 2D."


# Test: Image Type Processing
@pytest.mark.parametrize(
    "image_type, expected_func",
    [
        (mrd.ImageType.COMPLEX, lambda data: data.squeeze()),
        (mrd.ImageType.MAGNITUDE, lambda data: np.abs(data).squeeze()),
        (mrd.ImageType.PHASE, lambda data: np.angle(data).squeeze()),
        (mrd.ImageType.REAL, lambda data: data.squeeze().real),
        (mrd.ImageType.IMAG, lambda data: data.squeeze().imag),
    ],
)
def test_unsort_images_image_types(mock_image_array, image_type, expected_func):
    for header in mock_image_array.headers:
        header.image_type = image_type

    result = unsort_images(mock_image_array)

    for img, header in zip(result, mock_image_array.headers):
        expected_data = expected_func(
            mock_image_array.data[header.contrast, header.slice]
        )
        assert np.allclose(
            img.data, expected_data
        ), f"Data mismatch for image type {image_type}."


# Test: Empty ImageArray
def test_unsort_images_empty():
    mock_data = np.array([])
    mock_headers = []
    mock_meta = []
    empty_image_array = MockImageArray(
        data=mock_data, headers=mock_headers, meta=mock_meta
    )

    result = unsort_images(empty_image_array)
    assert result == [], "Output should be an empty list for empty input."
