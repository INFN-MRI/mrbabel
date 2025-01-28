"""Siemens to MRD Conversion Utilities."""

__all__ = [
    "read_siemens_header",
    "read_siemens_acquisitions",
]

import base64
import copy
import json
import warnings
import xml.etree.ElementTree as ET

from lxml import etree
from os.path import dirname
from os.path import join as pjoin

import numpy as np

import ismrmrd
import mrd

from twixtools import quat

from ._ismrmd2mrd import read_ismrmrd_header


def read_siemens_header(
    twix_obj: list[dict],
    hdr_template: mrd.Header | None = None,
    xml_file: str = None,
    xsl_file: str = None,
) -> mrd.Header:
    """Create MRD Header from a Siemens file."""
    headers_val = [
        _read_siemens_header(el["hdr"], xml_file, xsl_file) for el in twix_obj
    ]

    # find encoding map
    enc_map = {}
    reverse_enc_map = {}
    headers_keys = []
    for idx in range(len(twix_obj)):
        keys = list(twix_obj[idx].keys())
        if "hdr" in keys:
            keys.remove("hdr")
        if "hdr_str" in keys:
            keys.remove("hdr_str")
        headers_keys.append(keys[0])
        enc_map[idx] = keys[0]
        reverse_enc_map[keys[0]] = idx

    # get encodings
    headers = dict(zip(headers_keys, headers_val))
    encodings = []
    for idx in range(len(twix_obj)):
        key = enc_map[idx]
        if key != "image":  # make sure image is the last
            encodings.extend(headers[key].encoding)
    encodings.extend(headers["image"].encoding)

    # update header
    head = headers["image"]
    head.encoding = encodings
    head.user_parameters.user_parameter_double.append(
        mrd.UserParameterDoubleType(
            name="SliceThickness",
            value=twix_obj[reverse_enc_map["image"]]["hdr"]["Phoenix"]["sSliceArray"][
                "asSlice"
            ][0]["dThickness"],
        )
    )
    head.user_parameters.user_parameter_base64.append(
        mrd.UserParameterBase64Type(
            name="EncodingMap",
            value=base64.b64encode(json.dumps(enc_map).encode("utf-8")).decode("utf-8"),
        )
    )

    # update with blueprint
    if hdr_template is not None:
        # replace encoding
        head.encoding = hdr_template.encoding

        # replace contrast
        head.sequence_parameters = hdr_template.sequence_parameters

        # update user parameters
        head.user_parameters.extend(hdr_template.user_parameters)

    return head


def _read_siemens_header(twix_hdr: dict, xml_file: str, xsl_file: str) -> mrd.Header:
    """Create MRD Header from a Siemens file."""
    baseline_string = twix_hdr["Meas"]["tBaselineString"]

    # get version
    is_VB = (
        "VB17" in baseline_string
        or "VB15" in baseline_string
        or "VB13" in baseline_string
        or "VB11" in baseline_string
    )
    is_NX = "NVXA" in baseline_string or "syngo MR XA" in baseline_string

    # get converters
    if xml_file is None:
        if is_VB:
            xml_file = pjoin(
                dirname(__file__),
                "_siemens_pmaps",
                "IsmrmrdParameterMap_Siemens_VB17.xml",
            )
        else:
            xml_file = pjoin(
                dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens.xml"
            )
    if xsl_file is None:
        if is_NX:
            xsl_file = pjoin(
                dirname(__file__),
                "_siemens_pmaps",
                "IsmrmrdParameterMap_Siemens_NX.xsl",
            )
        else:
            xsl_file = pjoin(
                dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens.xsl"
            )

    # convert
    twix_xml = get_xml_from_siemens(twix_hdr, xml_file)
    ismrmrd_xml = convert_siemens_xml_to_ismrmrd_xml(twix_xml, xsl_file)

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

    return read_ismrmrd_header(ismrmrd_head)


def read_siemens_acquisitions(
    twix_obj, acquisitions_template=None
) -> list[mrd.Acquisition]:
    """Create a list of MRD Acquisitions from a list of Siemens Acquisitions."""
    nmeasurements = len(twix_obj)
    acquisitions = []
    for idx in range(nmeasurements):
        twix_obj[idx].pop("hdr_str", None)
        twix_hdr = twix_obj[idx].pop("hdr", None)
        twix_acquisitions = list(twix_obj[idx].values())[0].mdb_list

        # get acquisitions for current measurement
        nacquisitions = len(twix_acquisitions)
        acquisitions.extend(
            [
                read_siemens_acquisition(twix_acquisitions[n], twix_hdr, idx)
                for n in range(nacquisitions)
            ]
        )

    # update
    if acquisitions_template is not None:
        nacquisitions = len(acquisitions)
        for n in range(nacquisitions):
            acquisitions[n].head.flags = acquisitions_template[n].head.flags
            acquisitions[n].head.idx.kspace_encode_step_1 = acquisitions_template[
                n
            ].head.idx.kspace_encode_step_1
            acquisitions[n].head.idx.kspace_encode_step_2 = acquisitions_template[
                n
            ].head.idx.kspace_encode_step_2
            acquisitions[n].head.idx.slice = acquisitions_template[n].head.idx.slice
            acquisitions[n].head.idx.contrast = acquisitions_template[
                n
            ].head.idx.contrast
            acquisitions[n].head.discard_pre = acquisitions_template[n].head.discard_pre
            acquisitions[n].head.discard_post = acquisitions_template[
                n
            ].head.discard_post
            acquisitions[n].head.center_sample = acquisitions_template[
                n
            ].head.center_sample
            acquisitions[n].head.encoding_space_ref = acquisitions_template[
                n
            ].head.encoding_space_ref
            acquisitions[n].head.sample_time_us = acquisitions_template[
                n
            ].head.sample_time_us

    return acquisitions


def read_siemens_acquisition(twix_acquisition, twix_hdr, enc_ref) -> mrd.Acquisition:
    """Create MRD Acquisition from a Siemens Acquisition."""
    acquisition = mrd.Acquisition()

    # Fill in the header fields
    defs = mrd.AcquisitionFlags
    if twix_acquisition.mdh.EvalInfoMask & (1 << 25):
        acquisition.head.flags = defs.IS_NOISE_MEASUREMENT
    if twix_acquisition.mdh.EvalInfoMask & (1 << 28):
        acquisition.head.flags = defs.FIRST_IN_SLICE
    if twix_acquisition.mdh.EvalInfoMask & (1 << 29):
        acquisition.head.flags = defs.LAST_IN_SLICE
    if twix_acquisition.mdh.EvalInfoMask & (1 << 11):
        acquisition.head.flags = defs.LAST_IN_REPETITION

    # if a line is both image and ref, then do not set the ref flag
    if twix_acquisition.mdh.EvalInfoMask & (1 << 23):
        acquisition.head.flags = defs.IS_PARALLEL_CALIBRATION_AND_IMAGING
    else:
        if twix_acquisition.mdh.EvalInfoMask & (1 << 22):
            acquisition.head.flags = defs.IS_PARALLEL_CALIBRATION

    if twix_acquisition.mdh.EvalInfoMask & (1 << 24):
        acquisition.head.flags = defs.IS_REVERSE
    if twix_acquisition.mdh.EvalInfoMask & (1 << 11):
        acquisition.head.flags = defs.LAST_IN_MEASUREMENT
    if twix_acquisition.mdh.EvalInfoMask & (1 << 21):
        acquisition.head.flags = defs.IS_PHASECORR_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 1):
        acquisition.head.flags = defs.IS_NAVIGATION_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 1):
        acquisition.head.flags = defs.IS_RTFEEDBACK_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 2):
        acquisition.head.flags = defs.IS_HPFEEDBACK_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 51):
        acquisition.head.flags = defs.IS_DUMMYSCAN_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 10):
        acquisition.head.flags = defs.IS_SURFACECOILCORRECTIONSCAN_DATA
    if twix_acquisition.mdh.EvalInfoMask & (1 << 5):
        acquisition.head.flags = defs.IS_DUMMYSCAN_DATA

    if twix_acquisition.mdh.EvalInfoMask & (1 << 46):
        acquisition.head.flags = defs.LAST_IN_MEASUREMENT

    encoding_counter = mrd.EncodingCounters()
    encoding_counter.kspace_encode_step_1 = twix_acquisition.mdh.Counter.Lin
    encoding_counter.kspace_encode_step_2 = twix_acquisition.mdh.Counter.Par
    encoding_counter.average = twix_acquisition.mdh.Counter.Ave
    encoding_counter.slice = twix_acquisition.mdh.Counter.Sli
    encoding_counter.contrast = twix_acquisition.mdh.Counter.Eco
    encoding_counter.phase = twix_acquisition.mdh.Counter.Phs
    encoding_counter.repetition = twix_acquisition.mdh.Counter.Rep
    encoding_counter.set = twix_acquisition.mdh.Counter.Set
    encoding_counter.segment = twix_acquisition.mdh.Counter.Seg
    encoding_counter.user.extend(
        [
            twix_acquisition.mdh.Counter.Ida,
            twix_acquisition.mdh.Counter.Idb,
            twix_acquisition.mdh.Counter.Idc,
            twix_acquisition.mdh.Counter.Idd,
            twix_acquisition.mdh.Counter.Ide,
        ]
    )

    acquisition.head.idx = encoding_counter
    acquisition.head.measurement_uid = twix_acquisition.mdh.MeasUID
    acquisition.head.scan_counter = twix_acquisition.mdh.ScanCounter
    acquisition.head.acquisition_time_stamp = twix_acquisition.mdh.TimeStamp
    acquisition.head.physiology_time_stamp = twix_acquisition.mdh.PMUTimeStamp
    for n in range(twix_acquisition.mdh.UsedChannels):
        acquisition.head.channel_order.append(n)

    acquisition.head.discard_pre = int(twix_acquisition.mdh.CutOff.Pre)
    acquisition.head.discard_post = int(twix_acquisition.mdh.CutOff.Post)
    acquisition.head.center_sample = int(twix_acquisition.mdh.CenterCol)
    acquisition.head.encoding_space_ref = enc_ref
    acquisition.head.sample_time_us = (
        twix_hdr["MeasYaps"]["sRXSPEC"]["alDwellTime"][0] / 1000.0
    )

    position = [
        twix_acquisition.mdh.SliceData.SlicePos.Sag,
        twix_acquisition.mdh.SliceData.SlicePos.Cor,
        twix_acquisition.mdh.SliceData.SlicePos.Tra,
    ]
    acquisition.head.position = position

    quaternion = [
        twix_acquisition.mdh.SliceData.Quaternion[1],
        twix_acquisition.mdh.SliceData.Quaternion[2],
        twix_acquisition.mdh.SliceData.Quaternion[3],
        twix_acquisition.mdh.SliceData.Quaternion[0],
    ]
    read_dir, phase_dir, slice_dir = quat.quaternion_to_directions(quaternion)
    acquisition.head.read_dir = read_dir
    acquisition.head.phase_dir = phase_dir
    acquisition.head.slice_dir = slice_dir

    patient_table_position = [
        twix_acquisition.mdh.PTABPosX,
        twix_acquisition.mdh.PTABPosY,
        twix_acquisition.mdh.PTABPosZ,
    ]
    acquisition.head.patient_table_position = patient_table_position

    acquisition.head.user_int.extend(list(twix_acquisition.mdh.IceProgramPara[:7]))
    acquisition.head.user_int.append(twix_acquisition.mdh.TimeSinceLastRF)
    acquisition.head.user_float.extend(list(twix_acquisition.mdh.IceProgramPara[8:16]))

    # # Resize the data structure (for example, using numpy arrays or lists)
    acquisition.data = twix_acquisition.data

    return acquisition


# %% local utils
def get_xml_from_siemens(twix_hdr, xml_file):
    raw_head = _parse_hdr(copy.deepcopy(twix_hdr), xml_file)
    return _hdr2xml(raw_head, xml_file)


def convert_siemens_xml_to_ismrmrd_xml(twix_xml, xsl_file):
    """Apply an XSL transformation to an input XML string using an XSLT file."""
    # Parse the input XML and XSLT
    xml_tree = etree.fromstring(twix_xml)

    # Preprocess: Expand all lists in the XML tree
    xml_tree = _expand_all_as_sequence(xml_tree)

    # Parse XSLT
    xsl_tree = etree.parse(xsl_file)

    # Create an XSLT transformer
    transform = etree.XSLT(xsl_tree)

    # Apply the transformation
    result_tree = transform(xml_tree)

    return str(result_tree)


def _flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recurse into dictionaries
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Handle lists and tuples
            for idx, item in enumerate(v):
                indexed_key = f"{new_key}{sep}{idx}"
                if isinstance(item, dict):
                    # Recurse into dictionaries inside lists/tuples
                    items.extend(_flatten_dict(item, indexed_key, sep=sep).items())
                else:
                    items.append((indexed_key, item))
        else:
            # Base case: Add the key-value pair
            items.append((new_key, v))
    return dict(items)


def _merge_dicts(dicts):
    merged_dict = {}

    for d in dicts:
        for key, value in d.items():
            merged_dict[key] = value

    return merged_dict


def _parse_hdr(twix_hdr, xmlfile):
    twix_hdr.pop("Config_raw", None)
    twix_hdr.pop("Dicom_raw", None)
    twix_hdr.pop("Meas_raw", None)
    twix_hdr.pop("MeasYaps_raw", None)
    twix_hdr.pop("Phoenix_raw", None)
    twix_hdr.pop("Spice_raw", None)

    mappings = _parse_xml(xmlfile)
    _raw_hdr = {
        "Config": twix_hdr["Config"],
        "Dicom": twix_hdr["Dicom"],
        "Meas": twix_hdr["Meas"],
        "MeasYaps": _flatten_dict(twix_hdr["MeasYaps"]),
        "Phoenix": _flatten_dict(twix_hdr["Phoenix"]),
        "Spice": twix_hdr["Spice"],
    }
    _raw_hdr = _merge_dicts(list(_raw_hdr.values()))

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
    recon_dependencies = twix_hdr["Meas"]["ReconMeasDependencies"].split(" ")
    try:
        hdr["YAPS.ReconMeasDependencies.0"] = int(recon_dependencies[0])
    except Exception:
        pass
    try:
        hdr["YAPS.ReconMeasDependencies.1"] = int(recon_dependencies[1])
    except Exception:
        pass
    try:
        hdr["YAPS.ReconMeasDependencies.2"] = int(recon_dependencies[2])
    except Exception:
        pass
    # # manually fix IRIS and Header
    hdr["IRIS.DERIVED.phaseOversampling"] = twix_hdr["Meas"]["phaseOversampling"]
    hdr["IRIS.DERIVED.ImageColumns"] = twix_hdr["Meas"]["ImageColumns"]
    hdr["IRIS.DERIVED.ImageLines"] = twix_hdr["Meas"]["ImageLines"]
    hdr["IRIS.RECOMPOSE.PatientID"] = twix_hdr["Meas"]["PatientID"]
    hdr["IRIS.RECOMPOSE.PatientLOID"] = twix_hdr["Meas"]["PatientLOID"]
    hdr["IRIS.RECOMPOSE.PatientBirthDay"] = twix_hdr["Meas"]["PatientBirthDay"]
    hdr["IRIS.RECOMPOSE.StudyLOID"] = twix_hdr["Meas"]["StudyLOID"]
    hdr["HEADER.MeasUID"] = twix_hdr["Meas"]["MeasUID"]

    # # clean-up
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

    # clean contrasts
    ncontrasts = hdr["MEAS.lContrasts"]
    if "MEAS.alTR" in hdr:
        for n in range(ncontrasts, len(hdr["MEAS.alTR"])):
            hdr["MEAS.alTR"][n] = 0
    if "MEAS.alTE" in hdr:
        for n in range(ncontrasts, len(hdr["MEAS.alTE"])):
            hdr["MEAS.alTE"][n] = 0
    if "MEAS.alTI" in hdr:
        for n in range(ncontrasts, len(hdr["MEAS.alTI"])):
            hdr["MEAS.alTI"][n] = 0
    if "MEAS.adFlipAngleDegree" in hdr:
        for n in range(ncontrasts, len(hdr["MEAS.adFlipAngleDegree"])):
            hdr["MEAS.adFlipAngleDegree"] = 0

    return hdr


def _hdr2xml(raw_head, xmlfile):
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

    for n in range(64):
        new_dict["siemens"]["MEAS"]["asCoilSelectMeas"]["ID"]["tCoilID"].append(
            original_dict[
                f"MEAS.sCoilSelectMeas.aRxCoilSelectData.0.asList.{n}.sCoilElementID.tCoilID"
            ]
        )
        new_dict["siemens"]["MEAS"]["asCoilSelectMeas"]["Coil"]["lCoilCopy"].append(
            original_dict[
                f"MEAS.sCoilSelectMeas.aRxCoilSelectData.0.asList.{n}.sCoilElementID.lCoilCopy"
            ]
        )
        new_dict["siemens"]["MEAS"]["asCoilSelectMeas"]["Elem"]["tElement"].append(
            original_dict[
                f"MEAS.sCoilSelectMeas.aRxCoilSelectData.0.asList.{n}.sCoilElementID.tElement"
            ]
        )
        new_dict["siemens"]["MEAS"]["asCoilSelectMeas"]["Select"][
            "lElementSelected"
        ].append(
            original_dict[
                f"MEAS.sCoilSelectMeas.aRxCoilSelectData.0.asList.{n}.lElementSelected"
            ]
        )
        new_dict["siemens"]["MEAS"]["asCoilSelectMeas"]["Rx"][
            "lRxChannelConnected"
        ].append(
            original_dict[
                f"MEAS.sCoilSelectMeas.aRxCoilSelectData.0.asList.{n}.lRxChannelConnected"
            ]
        )
        new_dict["siemens"]["MEAS"]["asCoilSelectMeas"]["ADC"][
            "lADCChannelConnected"
        ].append(
            original_dict[
                f"MEAS.sCoilSelectMeas.aRxCoilSelectData.0.asList.{n}.lADCChannelConnected"
            ]
        )

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


def _expand_all_as_sequence(tree):
    root = tree  # In lxml, `tree` is the root element itself

    def expand_element(parent, element):
        # Check if the element's text is a vector-like value
        if element.text and element.text.startswith("[") and element.text.endswith("]"):
            # Parse the vector values
            values = element.text.strip("[]").split(",")
            values = [v.strip() for v in values]

            # Expand the element into multiple repeated elements
            for value in values:
                new_elem = etree.Element(element.tag)
                new_elem.text = value
                parent.append(new_elem)

            # Remove the original vector-valued element
            parent.remove(element)

    # Traverse and process the tree
    for parent in root.xpath(".//*"):  # XPath selects all elements in the tree
        for child in list(parent):
            expand_element(parent, child)

    return tree
