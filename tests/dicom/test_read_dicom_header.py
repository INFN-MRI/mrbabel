"""Test MRD Header parsing from DICOM."""

import pytest
import pydicom

from unittest.mock import MagicMock

from mrbabel.io.dicom._read import read_dicom_header


@pytest.fixture
def mock_dicom_dataset():
    """Fixture for creating a mock DICOM dataset."""
    dset = MagicMock(spec=pydicom.Dataset)

    # Mocking patient information
    dset.PatientName = "John Doe"
    dset.PatientWeight = 70.0
    dset.PatientHeight = 1.75
    dset.PatientID = "12345"
    dset.PatientBirthDate = "19800101"
    dset.PatientSex = "M"

    # Mocking study information
    dset.StudyDate = "20231206"
    dset.StudyTime = "123456"
    dset.StudyID = "67890"
    dset.StudyDescription = "Test Study"
    dset.StudyInstanceUID = "1.2.3.4.5.6"

    # Mocking measurement information
    dset.SeriesInstanceUID = "1.2.3.4.5.6.7"
    dset.PatientPosition = "HFS"
    dset.SeriesDescription = "Test Series"
    dset.FrameOfReferenceUID = "1.2.3.4.5.6.7.8"

    # Mocking acquisition system information
    dset.Manufacturer = "MockVendor"
    dset.ManufacturerModelName = "MockModel"
    dset.MagneticFieldStrength = 3.0
    dset.InstitutionName = "MockInstitution"
    dset.StationName = "MockStation"

    # Mocking imaging parameters
    dset.Columns = 256
    dset.Rows = 256
    dset.PixelSpacing = [0.5, 0.5]
    dset.SliceThickness = 1.0
    sop_class_uid = MagicMock()
    sop_class_uid.name = "MR Image Storage"
    dset.SOPClassUID = sop_class_uid
    
    # Mocking Sequence Parameters
    dset.FlipAngle = 10.0
    dset.RepetitionTime = 100.0
    dset.EchoTime = 5.0

    return dset


def test_read_dicom_header_basic(mock_dicom_dataset):
    """Test basic functionality of read_dicom_header."""
    mrd_header = read_dicom_header(mock_dicom_dataset)

    # Assert patient information
    assert mrd_header.subject_information.patient_name == "John Doe"
    assert mrd_header.subject_information.patient_weight_kg == 70.0
    assert mrd_header.subject_information.patient_height_m == 1.75
    assert mrd_header.subject_information.patient_id == "12345"
    assert mrd_header.subject_information.patient_birthdate == "19800101"
    assert mrd_header.subject_information.patient_gender.name == "M"

    # Assert study information
    assert mrd_header.study_information.study_date == "20231206"
    assert mrd_header.study_information.study_time == "123456"
    assert mrd_header.study_information.study_id == "67890"
    assert mrd_header.study_information.study_description == "Test Study"
    assert mrd_header.study_information.study_instance_uid == "1.2.3.4.5.6"

    # Assert measurement information
    assert mrd_header.measurement_information.measurement_id == "1.2.3.4.5.6.7"
    assert mrd_header.measurement_information.patient_position.name == "H_FS"
    assert mrd_header.measurement_information.series_description == "Test Series"
    assert (
        mrd_header.measurement_information.frame_of_reference_uid == "1.2.3.4.5.6.7.8"
    )

    # Assert acquisition system information
    assert mrd_header.acquisition_system_information.system_vendor == "MockVendor"
    assert mrd_header.acquisition_system_information.system_model == "MockModel"
    assert mrd_header.acquisition_system_information.system_field_strength_t == 3.0
    assert (
        mrd_header.acquisition_system_information.institution_name == "MockInstitution"
    )
    assert mrd_header.acquisition_system_information.station_name == "MockStation"

    # Assert encoding information
    assert mrd_header.encoding[0].encoded_space.matrix_size.x == 256
    assert mrd_header.encoding[0].encoded_space.matrix_size.y == 256
    assert mrd_header.encoding[0].encoded_space.matrix_size.z == 1
    assert mrd_header.encoding[0].encoded_space.field_of_view_mm.x == 128.0  # 0.5 * 256
    assert mrd_header.encoding[0].encoded_space.field_of_view_mm.y == 128.0
    assert mrd_header.encoding[0].encoded_space.field_of_view_mm.z == 1.0


def test_read_dicom_header_missing_optional_fields(mock_dicom_dataset):
    """Test handling of missing optional fields."""
    # Remove optional fields
    del mock_dicom_dataset.PatientHeight
    del mock_dicom_dataset.InstitutionName

    mrd_header = read_dicom_header(mock_dicom_dataset)

    # Height should be missing, InstitutionName should default to "Virtual"
    assert mrd_header.subject_information.patient_height_m is None
    assert mrd_header.acquisition_system_information.institution_name == "Virtual"


def test_read_dicom_header_enhanced_mr_storage(mock_dicom_dataset):
    """Test handling of Enhanced MR Image Storage SOP Class."""
    mock_dicom_dataset.SOPClassUID.name = "Enhanced MR Image Storage"

    # Mock enhanced MR-specific fields
    mock_dicom_dataset.PerFrameFunctionalGroupsSequence = [
        MagicMock(
            PixelMeasuresSequence=[
                MagicMock(SliceThickness=1.5, SpacingBetweenSlices=2.0)
            ]
        )
    ]

    mock_dicom_dataset.SharedFunctionalGroupsSequence = [
        MagicMock(
            MRModifierSequence=[
                MagicMock(
                    ParallelReductionFactorInPlane=2.0,
                    ParallelReductionFactorOutOfPlane=1.0,
                )
            ]
        )
    ]

    mrd_header = read_dicom_header(mock_dicom_dataset)

    assert mrd_header.user_parameters.user_parameter_double[0].name == "SliceThickness"
    assert mrd_header.user_parameters.user_parameter_double[0].value == 1.5
    assert (
        mrd_header.user_parameters.user_parameter_double[1].name
        == "SpacingBetweenSlices"
    )
    assert mrd_header.user_parameters.user_parameter_double[1].value == 2.0


def test_read_dicom_header_no_spacing_between_slices(mock_dicom_dataset):
    """Test handling when SpacingBetweenSlices is missing."""
    del mock_dicom_dataset.SpacingBetweenSlices

    with pytest.warns(
        UserWarning,
        match="Slice thickness and spacing info not found; assuming contiguous slices!",
    ):
        mrd_header = read_dicom_header(mock_dicom_dataset)

    assert (
        mrd_header.user_parameters.user_parameter_double[1].name
        == "SpacingBetweenSlices"
    )
    assert (
        mrd_header.user_parameters.user_parameter_double[1].value == 1.0
    )  # Defaulted to SliceThickness
