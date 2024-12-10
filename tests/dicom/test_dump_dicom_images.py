"""Test DICOM parsing from MRD Images."""

import pytest
import pydicom

import numpy as np

from unittest.mock import Mock
from mrbabel.io.dicom._mrd2dicom import dump_dicom_images, _dump_dicom_image


# Mock for MRD classes
class MockMRDImage:
    def __init__(self, data, head, meta):
        self.data = data
        self.head = head
        self.meta = meta


class MockMRDHeader:
    def __init__(
        self,
        patient_information=None,
        study_information=None,
        measurement_information=None,
        acquisition_system_information=None,
        experimental_conditions=None,
        sequence_parameters=None,
        encoding=None,
    ):
        self.patient_information = patient_information
        self.study_information = study_information
        self.measurement_information = measurement_information
        self.acquisition_system_information = acquisition_system_information
        self.experimental_conditions = experimental_conditions
        self.sequence_parameters = sequence_parameters
        self.encoding = encoding


@pytest.fixture
def mock_mrd_header():
    """Creates a mock MRD header with necessary fields."""
    return MockMRDHeader(
        patient_information=Mock(
            patient_name="John Doe",
            weight_kg=70,
            height_m=1.75,
            patient_id="12345",
            patient_birthdate="19800101",
            patient_gender="M",
        ),
        study_information=Mock(
            study_date="20240101",
            study_time="120000",
            study_id="STUDY123",
            study_description="Test Study",
            study_instance_uid="1.2.3.4.5",
        ),
        measurement_information=Mock(
            measurement_id="SERIES123",
            patient_position=Mock(name="HFS"),
            protocol_name="Test Protocol",
            frame_of_reference_uid="1.2.3.4.5.6",
        ),
        acquisition_system_information=Mock(
            system_vendor="VendorX",
            system_model="ModelY",
            system_field_strength_t=3.0,
            institution_name="Test Hospital",
            station_name="Test Station",
        ),
        experimental_conditions=Mock(h1resonance_frequency_hz=123456789.0),
        sequence_parameters=Mock(flip_angle_deg=30, t_r=2000, t_e=30, t_i=0),
        encoding=[
            Mock(
                encoded_space=Mock(
                    field_of_view_mm=Mock(z=240.0), matrix_size=Mock(z=24)
                )
            )
        ],
    )


@pytest.fixture
def mock_mrd_images():
    """Creates a list of mock MRD images."""
    images = []
    for i in range(5):
        data = np.random.rand(1, 1, 128, 128).astype(np.float32)
        head = Mock(
            image_series_index=1,
            image_index=i + 1,
            image_type="MAGNITUDE",
            field_of_view=[240.0, 240.0, 5.0],
            position=[0.0, 0.0, i * 5.0],
            read_dir=[1.0, 0.0, 0.0],
            phase_dir=[0.0, 1.0, 0.0],
            acquisition_time_stamp=1000 * (i + 1),
            physiology_time_stamp=[0],
        )
        meta = {"SeriesDescription": "Test Series", "ImageComment": [f"Slice {i}"]}
        images.append(MockMRDImage(data, head, meta))
    return images


# Test: dump_dicom_images
def test_dump_dicom_images(mock_mrd_images, mock_mrd_header):
    dicom_files = dump_dicom_images(mock_mrd_images, mock_mrd_header)
    assert len(dicom_files) == len(
        mock_mrd_images
    ), "Number of DICOM files should match number of input images."

    for dicom in dicom_files:
        assert isinstance(
            dicom, pydicom.dataset.Dataset
        ), "Each output should be a DICOM Dataset."
        assert (
            dicom.PatientName == mock_mrd_header.patient_information.patient_name
        ), "Patient name mismatch."
        assert dicom.Rows == 128, "Row count mismatch."
        assert dicom.Columns == 128, "Column count mismatch."


# Test: _dump_dicom_image
def test_dump_dicom_image(mock_mrd_images, mock_mrd_header):
    for image in mock_mrd_images:
        dicom = _dump_dicom_image(image, mock_mrd_header)
        assert isinstance(
            dicom, pydicom.dataset.Dataset
        ), "Output should be a DICOM Dataset."
        assert dicom.PixelData is not None, "PixelData should be populated."
        assert (
            dicom.PatientName == mock_mrd_header.patient_information.patient_name
        ), "Patient name mismatch."
        assert dicom.Rows == 128, "Row count mismatch."
        assert dicom.Columns == 128, "Column count mismatch."
