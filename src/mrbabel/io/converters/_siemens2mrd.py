"""Siemens to MRD Conversion Utilities."""

__all__ = [
    "read_siemens_header",
    "read_siemens_acquisitions",
]

import warnings
import xml.etree.ElementTree as ET

from lxml import etree
from os.path import dirname
from os.path import join as pjoin

import numpy as np

import ismrmrd
import mrd

from ._ismrmd2mrd import read_ismrmrd_header


def read_siemens_header(twixHdr: dict) -> mrd.Header:
    """Create MRD Header from a Siemens file."""
    baseLineString = twixHdr["Meas"]["tBaselineString"]

    # get version
    isVB = (
        "VB17" in baseLineString
        or "VB15" in baseLineString
        or "VB13" in baseLineString
        or "VB11" in baseLineString
    )
    isNX = "NVXA" in baseLineString or "syngo MR XA" in baseLineString

    # get converters
    if isVB:
        xmlfile = pjoin(
            dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens_VB17.xml"
        )
    else:
        xmlfile = pjoin(
            dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens.xml"
        )
    if isNX:
        xslfile = pjoin(
            dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens_NX.xsl"
        )
    else:
        xslfile = pjoin(
            dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens.xsl"
        )

    # convert
    raw_head = _parse_hdr(twixHdr, xmlfile)

    raw_xml = _head2xml(raw_head, xmlfile)
    ismrmrd_xml = _apply_xsl_transform(raw_xml, xslfile)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ismrmrd_head = ismrmrd.xsd.CreateFromDocument(ismrmrd_xml)

    # clean up user parameters
    ismrmrd_head.userParameters.userParameterBase64 = [
        param
        for param in ismrmrd_head.userParameters.userParameterBase64
        if (param.value != "" and param.value != "[]")
    ]
    ismrmrd_head.userParameters.userParameterDouble = [
        param
        for param in ismrmrd_head.userParameters.userParameterDouble
        if (param.value != "" and param.value != "[]")
    ]
    ismrmrd_head.userParameters.userParameterLong = [
        param
        for param in ismrmrd_head.userParameters.userParameterLong
        if (param.value != "" and param.value != "[]")
    ]
    ismrmrd_head.userParameters.userParameterString = [
        param
        for param in ismrmrd_head.userParameters.userParameterString
        if (param.value != "" and param.value != "[]")
    ]

    # clean sequence parameters
    for attr in vars(ismrmrd_head.sequenceParameters).keys():
        value = getattr(ismrmrd_head.sequenceParameters, attr)
        if isinstance(value, list):
            if np.isnan(value).any().item():
                setattr(ismrmrd_head.sequenceParameters, attr, [])

    # clean relative position
    if ismrmrd_head.measurementInformation.relativeTablePosition:
        invalid_pos = [
            el == ""
            for el in vars(
                ismrmrd_head.measurementInformation.relativeTablePosition
            ).values()
        ]
        if any(invalid_pos):
            ismrmrd_head.measurementInformation.relativeTablePosition = None

    return read_ismrmrd_header(ismrmrd_head), raw_head


def read_siemens_acquisitions(twixObj, twixHdr, twixAcquisitions, enc_ref=0) -> list[mrd.Acquisition]:
    """Create a list of MRD Acquisitions from a list of ISMRMRD Acquisitions."""
    twixAcquisitions.removeOS = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        twixData = np.ascontiguousarray(twixAcquisitions.unsorted().T)
    nacquisitions = twixData.shape[0]
    return [
        read_siemens_acquisition(twixHdr, twixAcquisitions, twixData, n, enc_ref)
        for n in range(nacquisitions)
    ]


def read_siemens_acquisition(twixHdr, twixAcquisitions, twixData, n, enc_ref) -> mrd.Acquisition:
    """Create MRD Acquisition from a ISMRMRD Acquisition."""
    acquisition = mrd.Acquisition()

    # Fill in the header fields
    # acquisition.head.flags = acq.flags

    encoding_counter = mrd.EncodingCounters()
    encoding_counter.kspace_encode_step_1 = int(twixAcquisitions.Lin[n])
    encoding_counter.kspace_encode_step_2 = int(twixAcquisitions.Par[n])
    encoding_counter.average = int(twixAcquisitions.Ave[n])
    encoding_counter.slice = int(twixAcquisitions.Sli[n])
    encoding_counter.contrast = int(twixAcquisitions.Eco[n])
    encoding_counter.phase = int(twixAcquisitions.Phs[n])
    encoding_counter.repetition = int(twixAcquisitions.Rep[n])
    encoding_counter.set = int(twixAcquisitions.Set[n])
    encoding_counter.segment = int(twixAcquisitions.Seg[n])
    encoding_counter.user.extend(
        [
            int(twixAcquisitions.Ida[n]),
            int(twixAcquisitions.Idb[n]),
            int(twixAcquisitions.Idc[n]),
            int(twixAcquisitions.Idd[n]),
            int(twixAcquisitions.Ide[n]),
        ]
    )

    acquisition.head.idx = encoding_counter
    acquisition.head.measurement_uid = int(twixHdr.Meas.MeasUID)
    acquisition.head.scan_counter = int(twixAcquisitions.scancounter[n])
    acquisition.head.acquisition_time_stamp = float(twixAcquisitions.timestamp[n])
    acquisition.head.physiology_time_stamp = float(twixAcquisitions.pmutime[n])
    for n in range(twixAcquisitions.NCha):
        acquisition.head.channel_order.append(n)

    acquisition.head.discard_pre = int(twixAcquisitions.cutOff[n][0])
    acquisition.head.discard_post = int(twixAcquisitions.cutOff[n][1])
    acquisition.head.center_sample = int(twixAcquisitions.centerCol[n])
    acquisition.head.encoding_space_ref = enc_ref
    acquisition.head.sample_time_us = twixHdr.MeasYaps[('sRXSPEC','alDwellTime', '0')] / 1000.0

    # acquisition.head.position = list(acq.position)
    # acquisition.head.read_dir = list(acq.read_dir)
    # acquisition.head.phase_dir = list(acq.phase_dir)
    # acquisition.head.slice_dir = list(acq.slice_dir)
    # acquisition.head.patient_table_position = list(acq.patient_table_position)

    # acquisition.head.user_int.extend(list(acq.user_int))
    # acquisition.head.user_float.extend(list(acq.user_float))

    # # Resize the data structure (for example, using numpy arrays or lists)
    acquisition.data = twixData[n]

    # # If trajectory dimensions are present, resize and fill the trajectory data
    # if acq.trajectory_dimensions > 0:
    #     acquisition.trajectory = acq.traj

    return acquisition


# %% local utils
def _merge_and_convert_dicts(dicts):
    merged_dict = {}

    for d in dicts:
        for key, value in d.items():
            # Convert tuple keys into dotted string
            if isinstance(key, tuple):
                key = ".".join(key)
            # Merge into the final dictionary
            merged_dict[key] = value

    return merged_dict


def _parse_hdr(twixHdr, xmlfile):
    mappings = _parse_xml(xmlfile)
    _raw_hdr = _merge_and_convert_dicts(list(twixHdr.values()))

    fields = list(mappings.keys())
    hdr = {}
    for field in fields:
        key = ".".join(field.split(".")[1:])
        if key in _raw_hdr:
            value = _raw_hdr[key]
        else:
            value = []

        hdr[field] = value

    # manually fix Dwell Time
    hdr["MEAS.sRXSPEC.alDwellTime"] = _raw_hdr["sRXSPEC.alDwellTime.0"]

    # manually fix IRIS and Header
    hdr["IRIS.DERIVED.phaseOversampling"] = twixHdr.Meas.phaseOversampling
    hdr["IRIS.DERIVED.ImageColumns"] = twixHdr.Meas.ImageColumns
    hdr["IRIS.DERIVED.ImageLines"] = twixHdr.Meas.ImageLines
    hdr["IRIS.RECOMPOSE.PatientID"] = twixHdr.Meas.PatientID
    hdr["IRIS.RECOMPOSE.PatientLOID"] = twixHdr.Meas.PatientLOID
    hdr["IRIS.RECOMPOSE.PatientBirthDay"] = twixHdr.Meas.PatientBirthDay
    hdr["IRIS.RECOMPOSE.StudyLOID"] = twixHdr.Meas.StudyLOID
    hdr["HEADER.MeasUID"] = twixHdr.Meas.MeasUID

    # clean-up
    if "YAPS.iMaxNoOfRxChannels" in hdr:
        hdr["YAPS.iMaxNoOfRxChannels"] = int(hdr["YAPS.iMaxNoOfRxChannels"])
    if "DICOM.lFrequency" in hdr:
        hdr["DICOM.lFrequency"] = int(hdr["DICOM.lFrequency"])
    if "IRIS.DERIVED.ImageColumns" in hdr:
        hdr["IRIS.DERIVED.ImageColumns"] = int(hdr["IRIS.DERIVED.ImageColumns"])
    if "IRIS.DERIVED.ImageLines" in hdr:
        hdr["IRIS.DERIVED.ImageLines"] = int(hdr["IRIS.DERIVED.ImageLines"])
    if "MEAS.sKSpace.lImagesPerSlab" in hdr:
        hdr["MEAS.sKSpace.lImagesPerSlab"] = int(hdr["MEAS.sKSpace.lImagesPerSlab"])
    if "YAPS.iNoOfFourierColumns" in hdr:
        hdr["YAPS.iNoOfFourierColumns"] = int(hdr["YAPS.iNoOfFourierColumns"])
    if "YAPS.iPEFTLength" in hdr:
        hdr["YAPS.iPEFTLength"] = int(hdr["YAPS.iPEFTLength"])
    if "YAPS.i3DFTLength" in hdr:
        hdr["YAPS.i3DFTLength"] = int(hdr["YAPS.i3DFTLength"])
    if "MEAS.sPat.lAccelFactPE" in hdr:
        hdr["MEAS.sPat.lAccelFactPE"] = int(hdr["MEAS.sPat.lAccelFactPE"])
    if "MEAS.sPat.lAccelFact3D" in hdr:
        hdr["MEAS.sPat.lAccelFact3D"] = int(hdr["MEAS.sPat.lAccelFact3D"])
    if "MEAS.sSliceArray.asSlice.0.dReadoutFOV" in hdr:
        hdr["MEAS.sSliceArray.asSlice.0.dReadoutFOV"] = hdr[
            "MEAS.sSliceArray.asSlice.0.dReadoutFOV"
        ]
    if "MEAS.sSliceArray.asSlice.0.dPhaseFOV" in hdr:
        hdr["MEAS.sSliceArray.asSlice.0.dPhaseFOV"] = hdr[
            "MEAS.sSliceArray.asSlice.0.dPhaseFOV"
        ]
    if "MEAS.sSliceArray.asSlice.0.dThickness" in hdr:
        hdr["MEAS.sSliceArray.asSlice.0.dThickness"] = hdr[
            "MEAS.sSliceArray.asSlice.0.dThickness"
        ]

    return hdr


def _head2xml(raw_head, xmlfile):
    mappings = _parse_xml(xmlfile)
    output = _dict_to_xml("siemens", _transform_dict(raw_head, mappings)["siemens"])
    return ET.tostring(output, encoding="utf-8", method="xml")


def _parse_xml(xml_file):
    mappings = {}
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Iterate through <p> elements and extract <s> and <d> values
    for param in root.findall(".//p"):
        source = param.find("s").text.strip()
        destination = param.find("d").text.strip()
        mappings[source] = destination

    # simpler
    return mappings


def _transform_dict(original_dict, mappings):
    new_dict = {}
    for src_key, dst_key in mappings.items():
        # Check if the source key exists in the original dictionary
        if src_key in original_dict:
            # Split the destination key path and construct nested dictionary
            keys = dst_key.split(".")
            temp = new_dict
            for key in keys[:-1]:
                temp = temp.setdefault(key, {})
            temp[keys[-1]] = original_dict[src_key]
    return new_dict


def _dict_to_xml(tag, d):
    """Convert a dictionary to an XML element."""
    elem = ET.Element(tag)
    for key, val in d.items():
        if isinstance(val, dict):
            # Recursively handle nested dictionaries
            child = _dict_to_xml(key, val)
            elem.append(child)
        else:
            # Add simple key-value pairs as elements
            child = ET.SubElement(elem, key)
            child.text = str(val)
    return elem


def _apply_xsl_transform(input_xml, xsl_file):
    """Apply an XSL transformation to an input XML string using an XSLT file."""
    # Parse the input XML and XSLT
    xml_tree = etree.fromstring(input_xml)

    # Preprocess: Expand all lists in the XML tree
    _expand_all_lists(xml_tree)

    # Parse XSLT
    xsl_tree = etree.parse(xsl_file)

    # Create an XSLT transformer
    transform = etree.XSLT(xsl_tree)

    # Apply the transformation
    result_tree = transform(xml_tree)

    return str(result_tree)


def _expand_all_lists(xml_tree):
    """
    Expand all elements in the XML tree that represent lists into multiple elements.
    Specifically, converts space-separated string values to lists of elements.
    """
    for element in xml_tree.xpath("//*"):  # Iterate through all elements
        if element.text:
            # Check if the element text is a space-delimited string
            split_text = element.text.strip().split()
            if len(split_text) > 1:  # It's a space-separated string (list)
                parent = element.getparent()
                tag = element.tag

                # Remove the original element from the parent
                parent.remove(element)

                # Create new elements for each list item and append them to the parent
                for value in split_text:
                    new_element = etree.Element(tag)
                    new_element.text = value  # Set the value for each new element
                    parent.append(new_element)


# %% 
# Copied from https://github.com/pehses/twixtools
# helper functions to convert quaternions to read/phase/slice normal vectors
# and vice vectors
# direct translation from ismrmrd.c from the ismrmrd project

def quaternion_to_directions(quat):
    a, b, c, d = quat

    read_dir = 3 * [None]
    phase_dir = 3 * [None]
    slice_dir = 3 * [None]

    read_dir[0] = 1. - 2. * (b * b + c * c)
    phase_dir[0] = 2. * (a * b - c * d)
    slice_dir[0] = 2. * (a * c + b * d)

    read_dir[1] = 2. * (a * b + c * d)
    phase_dir[1] = 1. - 2. * (a * a + c * c)
    slice_dir[1] = 2. * (b * c - a * d)

    read_dir[2] = 2. * (a * c - b * d)
    phase_dir[2] = 2. * (b * c + a * d)
    slice_dir[2] = 1. - 2. * (a * a + b * b)

    return read_dir, phase_dir, slice_dir


def directions_to_quaternion(read_dir, phase_dir, slice_dir):

    r11, r21, r31 = read_dir
    r12, r22, r32 = phase_dir
    r13, r23, r33 = slice_dir

    a, b, c, d, s = 1, 0, 0, 0, 0
    trace = 0

    # verify the sign of the rotation
    if __sign_of_directions(read_dir, phase_dir, slice_dir) < 0:
        # flip 3rd column
        r13, r23, r33 = -r13, -r23, -r33

    # Compute quaternion parameters
    # http://www.cs.princeton.edu/~gewang/projects/darth/stuff/quat_faq.html#Q55
    trace = 1.0 + r11 + r22 + r33
    if trace > 0.00001:  # simplest case
        s = np.sqrt(trace) * 2
        a = (r32 - r23) / s
        b = (r13 - r31) / s
        c = (r21 - r12) / s
        d = 0.25 * s
    else:
        # trickier case...
        # determine which major diagonal element has
        # the greatest value...
        xd = 1.0 + r11 - (r22 + r33)  # 4**b**b
        yd = 1.0 + r22 - (r11 + r33)  # 4**c**c
        zd = 1.0 + r33 - (r11 + r22)  # 4**d**d
        # if r11 is the greatest
        if xd > 1.0:
            s = 2.0 * np.sqrt(xd)
            a = 0.25 * s
            b = (r21 + r12) / s
            c = (r31 + r13) / s
            d = (r32 - r23) / s
        # else if r22 is the greatest
        elif yd > 1.0:
            s = 2.0 * np.sqrt(yd)
            a = (r21 + r12) / s
            b = 0.25 * s
            c = (r32 + r23) / s
            d = (r13 - r31) / s
        # else, r33 must be the greatest
        else:
            s = 2.0 * np.sqrt(zd)
            a = (r13 + r31) / s
            b = (r23 + r32) / s
            c = 0.25 * s
            d = (r21 - r12) / s

        if a < 0.0:
            a, b, c, d = -a, -b, -c, -d

    return [a, b, c, d]


def __sign_of_directions(read_dir, phase_dir, slice_dir):
    r11, r21, r31 = read_dir
    r12, r22, r32 = phase_dir
    r13, r23, r33 = slice_dir

    # Determinant should be 1 or -1
    deti = (r11 * r22 * r33) + (r12 * r23 * r31) + (r21 * r32 * r13) -\
           (r13 * r22 * r31) - (r12 * r21 * r33) - (r11 * r23 * r32)

    if deti < 0:
        return -1
    else:
        return 1
    
internal_os = 2
pcs_directions = ["dSag", "dCor", "dTra"]

# p. 418 - pcs to dcs
pcs_transformations = {
    "HFS": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    "HFP": [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    "FFS": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
}


class Geometry:
    """Get geometric information from twix dict

    During initialization, information about slice geometry is copied from the supplied twix dict.
    Methods for conversion between the different coordinate systems
    Patient Coordinate System (PCS; Sag/Cor/Tra), Device Coordinate System (XYZ) and Gradient Coordinate System
    (GCS or PRS; Phase, Readout, Slice) are implemented (so far only rotation, i.e. won't work for offcenter measurementes).

    Examples
    ----------
    ```
    import twixtools
    twix = twixtools.read_twix('meas.dat', parse_geometry=True, parse_data=False)
    x = [1,1,1]
    y = twix[-1]['geometry'].rps_to_xyz() @ x
    ```

    Based on work from Christian Mirkes and Ali Aghaeifar.
    """

    def __init__(self, twix):
        self.from_twix(twix)

    def __str__(self):
        return ("Geometry:\n"
                f"  inplane_rot: {self.inplane_rot}\n"
                f"  normal: {self.normal}\n"
                f"  offset: {self.offset}\n"
                f"  patient_position: {self.patient_position}\n"
                f"  rotmatrix: {self.rotmatrix}\n"
                f"  voxelsize: {self.voxelsize}")

    def from_twix(self, twix):
        if twix["hdr"]["MeasYaps"][("sKSpace", "ucDimension")] == 2:
            self.dims = 2
        elif twix["hdr"]["MeasYaps"][("sKSpace", "ucDimension")] == 4:
            self.dims = 3
        else:
            self.dims = None

        if len(twix["hdr"]["MeasYaps"][("sKSpace", "asSlice")]) > 1:
            print("WARNING: Geometry calculations are valid only for the first slice in this multi-slice acquisition.")

        self.fov = [
            twix["hdr"]["MeasYaps"][("sKSpace", "asSlice", "0")]["dReadoutFOV"]
            * internal_os,
            twix["hdr"]["MeasYaps"][("sKSpace", "asSlice", "0")]["dPhaseFOV"],
            twix["hdr"]["MeasYaps"][("sKSpace", "asSlice", "0")]["dThickness"],
        ]

        self.resolution = [
            twix["hdr"]["MeasYaps"][("sKSpace", "lBaseResolution")] * internal_os,
            twix["hdr"]["MeasYaps"][("sKSpace", "lPhaseEncodingLines")],
            twix["hdr"]["MeasYaps"][("sKSpace", "lPartitions")] if self.dims == 3 else 1,
        ]

        self.voxelsize = list(np.array(self.fov) / np.array(self.resolution))

        self.normal = [0, 0, 0]
        if "sNormal" in twix["hdr"][("sKSpace", "asSlice", "0")]:
            for i, d in enumerate(pcs_directions):
                self.normal[i] = twix["hdr"]["MeasYaps"][("sKSpace", "asSlice", "0")][
                    "sNormal"
                ].get(d, self.normal[i])

        self.offset = [0, 0, 0]
        if "sPosition" in twix["hdr"]["MeasYaps"][("sKSpace", "asSlice", "0")]:
            for i, d in enumerate(pcs_directions):
                self.offset[i] = twix["hdr"]["MeasYaps"][("sKSpace", "asSlice", "0")][
                    "sPosition"
                ].get(d, self.offset[i])

        self.inplane_rot = twix["hdr"]["MeasYaps"][("sKSpace", "asSlice", "0")].get(
            "dInPlaneRot", 0
        )

        if "tPatientPosition" in twix["hdr"]["Meas"]:
            self.patient_position = twix["hdr"]["Meas"].get("tPatientPosition")
        elif "sPatPosition" in twix["hdr"]["Meas"]:
            self.patient_position = twix["hdr"]["Meas"].get("sPatPosition")
        else:
            self.patient_position = None

        self.rotmatrix = self.rps_to_xyz().tolist()

    def get_plane_orientation(self):
        # sanity check if normal vector is unit vector
        norm = np.linalg.norm(self.normal)
        if not abs(1 - norm) < 0.001:
            raise RuntimeError(f"Normal vector is not normal: |x| = {norm}")

        # find main direction of normal vector for first part of rot matrix
        maindir = np.argmax(np.abs(self.normal))
        if maindir == 0:
            init_mat = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # @ mat // inplane mat
        elif maindir == 1:
            init_mat = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        else:
            init_mat = np.eye(3)

        # initialize normal vector direction to which to compute the second part of rotation matrix
        init_normal = np.zeros(3)
        init_normal[maindir] = 1

        # calculate cross product and sine, cosine
        v = np.cross(init_normal, self.normal)
        s = np.linalg.norm(v)
        c = np.dot(init_normal, self.normal)

        if s <= 0.00001:
            # we have cosine 1 or -1, two vectors are (anti-) parallel
            mat = np.matmul(np.eye(3) * c, init_mat)
        else:
            # calculate cross product matrix
            v_x = np.cross(np.eye(3), v)
            # calculate rotation matrix, division should be possible from excluding c = -1 above
            mat = np.eye(3) + v_x + np.divide(np.matmul(v_x, v_x), 1 + c)
            # calculate full rotation matrix
            mat = np.matmul(mat, init_mat)

        return mat

    def get_inplane_rotation(self):
        mat = [
            [-np.sin(self.inplane_rot), np.cos(self.inplane_rot), 0],
            [-np.cos(self.inplane_rot), -np.sin(self.inplane_rot), 0],
            [0, 0, 1],
        ]
        return np.array(mat)

    def prs_to_pcs(self):
        mat = self.get_inplane_rotation()
        mat = self.get_plane_orientation() @ mat
        return mat

    def pcs_to_xyz(self):
        if self.patient_position in pcs_transformations:
            return np.array(pcs_transformations[self.patient_position])
        else:
            raise RuntimeError(f"Unknown patient position: {self.patient_position}")

    def prs_to_xyz(self):
        return self.pcs_to_xyz() @ self.prs_to_pcs()

    def rps_to_xyz(self):
        return self.prs_to_xyz() @ self.rps_to_prs()

    def rps_to_prs(self):
        return np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])