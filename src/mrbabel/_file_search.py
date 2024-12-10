"""Utilies for file search."""

__all__ = ["get_paths"]

import os
import glob
from typing import List, Union


def get_paths(ext: str, parent: Union[str, List[str]], ext2: str = "") -> List[str]:
    """
    Recursively find all files with a specific extension in the given folder(s) or matching a folder path pattern.

    Parameters
    ----------
    ext : str
        The file extension to search for (e.g., "nii").
    parent : str or list of str
        A folder path, a list of folder paths, or a path pattern (e.g., "my_folder/*").
    ext2 : str, optional
        Alternative file extension to search for (e.g., "nii.gz").

    Returns
    -------
    list of str
        A flattened list of absolute file paths matching the specified extension.

    Examples
    --------
    Find all `.dcm` files in a single folder:
    >>> get_paths(ext="dcm", parent="my_folder")
    ["abs-path-to-myfolder/file1.dcm", "abs-path-to-myfolder/file2.dcm"]

    Find all `.dcm` files in multiple folders:
    >>> get_paths(ext="dcm", parent=["my_folder1", "my_folder2"])
    ["abs-path-to-myfolder1/file1.dcm", ..., "abs-path-to-myfolder2/file2.dcm"]

    Find all `.dcm` files in folders matching a path pattern:
    >>> get_paths(ext="dcm", parent="base_path/my_folder*")
    ["abs-base_path/myfolder1/file1.dcm", ..., "abs-base_path/myfolderN/fileM.dcm"]
    """
    # Ensure `parent` is a list for uniform processing
    if isinstance(parent, str):
        parent = [parent]

    # Flatten list of all matched paths
    all_paths = []
    for p in parent:
        # Expand patterns like "my_folder/*" and normalize directory names
        matched_folders = glob.glob(p)
        for folder in matched_folders:
            folder = os.path.normpath(
                folder
            )  # Normalize paths (e.g., remove trailing slashes)
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith(f".{ext}") or file.endswith(f".{ext2}"):
                        all_paths.append(os.path.abspath(os.path.join(root, file)))

    return sorted(all_paths)
