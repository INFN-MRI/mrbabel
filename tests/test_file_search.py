"""Test file search utilities."""

import os
import tempfile
import pytest

from mrbabel._file_search import get_paths


@pytest.fixture
def create_files():
    """
    Pytest fixture to create a set of files for testing using a temporary directory.

    Yields
    ------
    Callable[[list of str], str]
        A function to create the specified files in a temporary directory.
        The function returns the path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:

        def _create_files(file_paths):
            for file in file_paths:
                file_path = os.path.join(temp_dir, file)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                open(file_path, "a").close()  # Create the file
            return temp_dir  # Return the temporary directory path

        yield _create_files  # Yield the helper function for use in tests


def test_get_paths_single_folder(create_files):
    """Test get_paths with a single folder."""
    base_path = create_files(["file1.dcm", "file2.dcm", "file3.txt"])
    results = get_paths(ext="dcm", parent=base_path)
    expected = [
        os.path.join(base_path, "file1.dcm"),
        os.path.join(base_path, "file2.dcm"),
    ]
    assert sorted(results) == sorted(expected)


def test_get_paths_multiple_folders(create_files):
    """Test get_paths with multiple folders."""
    base_path = create_files(
        [
            "folder1/file1.dcm",
            "folder1/file2.dcm",
            "folder2/file3.dcm",
            "folder2/file4.txt",
        ]
    )
    results = get_paths(
        ext="dcm",
        parent=[
            os.path.join(base_path, "folder1"),
            os.path.join(base_path, "folder2"),
        ],
    )
    expected = [
        os.path.join(base_path, "folder1/file1.dcm"),
        os.path.join(base_path, "folder1/file2.dcm"),
        os.path.join(base_path, "folder2/file3.dcm"),
    ]
    assert sorted(results) == sorted(expected)


def test_get_paths_with_pattern(create_files):
    """Test get_paths with a folder path pattern."""
    base_path = create_files(
        [
            "folder1/file1.dcm",
            "folder2/file2.dcm",
        ]
    )
    results = get_paths(ext="dcm", parent=os.path.join(base_path, "folder*"))
    expected = [
        os.path.join(base_path, "folder1/file1.dcm"),
        os.path.join(base_path, "folder2/file2.dcm"),
    ]
    assert sorted(results) == sorted(expected)


def test_get_paths_no_matching_files(create_files):
    """Test get_paths when no files match the extension."""
    base_path = create_files(["file1.txt", "file2.csv"])
    results = get_paths(ext="dcm", parent=base_path)
    assert results == []


def test_get_paths_with_trailing_slash(create_files):
    """Test get_paths with a folder path that includes a trailing slash."""
    base_path = create_files(["folder/file1.dcm", "folder/file2.txt"])
    folder = os.path.join(base_path, "folder")
    results = get_paths(ext="dcm", parent=folder + os.sep)
    expected = [
        os.path.join(folder, "file1.dcm"),
    ]
    assert sorted(results) == sorted(expected)


def test_get_paths_nested_structure(create_files):
    """Test get_paths with nested folder structures."""
    base_path = create_files(
        [
            "subfolder1/file1.dcm",
            "subfolder1/file2.txt",
            "subfolder2/subsubfolder/file3.dcm",
        ]
    )
    results = get_paths(ext="dcm", parent=base_path)
    expected = [
        os.path.join(base_path, "subfolder1/file1.dcm"),
        os.path.join(base_path, "subfolder2/subsubfolder/file3.dcm"),
    ]
    assert sorted(results) == sorted(expected)
