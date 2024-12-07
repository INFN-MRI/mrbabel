=========================================
mrbabel: MRI Data Reading and Sorting
=========================================

**mrbabel** is a Python package for reading, processing, and writing MRI data in various formats. The package supports both k-space and image-space data, enabling seamless integration with common acquisition and reconstruction pipelines.

Key Features
============

- **Read and Write Multiple Formats**:
  - K-space data: ISMRMRD/MRD, Siemens, GE
  - Image data: DICOM, NIfTI
- **Unsorted and Sorted Data Handling**:
  - Unsorted: acquisition-like streams
  - Sorted: organized arrays ready for reconstruction
- **Serialization**:
  - Convert between formats for both k-space and image data
- **Metadata Handling**:
  - Global headers and per-readout/image headers
  - Format-specific metadata parsing
- **Interoperability**:
  - Vendor-specific parsing and conversion utilities

Installation
============

You can install **mrbabel** using pip:

.. code-block:: bash

    pip install mrbabel

Requirements
============

- Python >= 3.10
- `numpy`
- `pydicom`
- `nibabel`
- `ismrmrd`

Documentation
=============

The full documentation is available at: `<https://github.com/INFN-MRI/mrbabel>`_

Quickstart
==========

Here’s a quick example of how to use **mrbabel**:

1. **Read k-space data** from an MRD file:

   .. code-block:: python

      from mrbabel.io import read_mrd_acquisitions

      header, acquisitions = read_mrd_acquisitions("data.mrd")

2. **Sort the acquisitions** for reconstruction:

   .. code-block:: python

      from mrbabel.data import sort_acquisitions

      sorted_data, sorted_traj, sorted_headers = sort_acquisitions(acquisitions)

3. **Write the sorted data** back to an MRD file:

   .. code-block:: python

      from mrbabel.io import write_mrd

      write_mrd("sorted_data.mrd", acquisitions, header)

Contributing
============

Contributions are welcome! Please feel free to submit issues or pull requests on our GitHub page.

License
=======

This project is licensed under the MIT License. See the LICENSE file for details.

Credits
=======

**mrbabel** is developed and maintained by:

- *Matteo Cencini* (Author/Maintainer)

Acknowledgments
===============

This project builds on tools and ideas from:

- ismrmrd
- mrd::python
- pydicom
- nibabel
- pyvoxel
- pymapvbvd

