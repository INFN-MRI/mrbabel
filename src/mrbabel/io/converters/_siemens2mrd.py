"""Siemens to MRD Conversion Utilities."""

__all__ = [
    "read_siemens_header",
]

import xml.etree.ElementTree as ET

from lxml import etree
from os.path import dirname
from os.path import join as pjoin

import ismrmrd
import mrd

from ._ismrmd2mrd import read_ismrmrd_header


def read_siemens_header(twixHdr: dict) -> mrd.Header:
    """Create MRD Header from a Siemens file."""
    baseLineString = twixHdr["Meas"]["tBaselineString"]
    
    # get version
    isVB = "VB17" in baseLineString or "VB15" in baseLineString or "VB13" in baseLineString or "VB11" in baseLineString
    isNX = "NVXA" in baseLineString or "syngo MR XA" in baseLineString
    
    # get converters
    if isVB:
        xmlfile = pjoin(dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens_VB17.xml")
    else:
        xmlfile = pjoin(dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens.xml")
    if isNX:
        xslfile = pjoin(dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens_NX.xsl")
    else:
        xslfile = pjoin(dirname(__file__), "_siemens_pmaps", "IsmrmrdParameterMap_Siemens.xsl")
        
    # convert
    raw_head = _flatten_dict(twixHdr)
    
    # some empirical fix?
    if "iMaxNoOfRxChannels" in raw_head:
        raw_head["iMaxNoOfRxChannels"] = int(raw_head["iMaxNoOfRxChannels"])
    if "lFrequency" in raw_head:
        raw_head["lFrequency"] = int(raw_head["lFrequency"])
    if "ImageColumns" in raw_head:
        raw_head["ImageColumns"] = int(raw_head["ImageColumns"])
    if "ImageLines" in raw_head:
        raw_head["ImageLines"] = int(raw_head["ImageLines"])
    if "lImagesPerSlab" in raw_head:
        raw_head["lImagesPerSlab"] = int(raw_head["lImagesPerSlab"])
    if "iNoOfFourierColumns" in raw_head:
        raw_head["iNoOfFourierColumns"] = int(raw_head["iNoOfFourierColumns"])
    if "iPEFTLength" in raw_head:
        raw_head["iPEFTLength"] = int(raw_head["iPEFTLength"])
    if "i3DFTLength" in raw_head:
        raw_head["i3DFTLength"] = int(raw_head["i3DFTLength"])
    if "lAccelFactPE" in raw_head:
        raw_head["lAccelFactPE"] = int(raw_head["lAccelFactPE"])
    if "lAccelFact3D" in raw_head:
        raw_head["lAccelFact3D"] = int(raw_head["lAccelFact3D"])
    if "ReadFoV" in raw_head:
        raw_head["dReadoutFOV"] = raw_head["ReadFoV"]
    if "PhaseFoV" in raw_head:
        raw_head["dPhaseFOV"] = raw_head["PhaseFoV"]
    if "thickness" in raw_head:
        raw_head["dThickness"] = raw_head["dThickness"]
    if "alTR" in raw_head:
        raw_head["alTR"] = float(raw_head["alTR"].split(" ")[0])
    if "alTE" in raw_head:
        raw_head["alTE"] = float(raw_head["alTE"].split(" ")[0])
    if "alTI" in raw_head:
        raw_head["alTI"] = float(raw_head["alTI"].split(" ")[0])
    raw_head.pop("alFree", None)
        
    raw_xml = _head2xml(raw_head, xmlfile)
    ismrmrd_xml = _apply_xsl_transform(raw_xml, xslfile)
    ismrmrd_head = ismrmrd.xsd.CreateFromDocument(ismrmrd_xml)
    return ismrmrd_head
    # return read_ismrmrd_header(ismrmrd_head)

def _flatten_dict(hdr):
    output = {}
    for field, subdict in hdr.items():
        for subfield, value in subdict.items():
            if isinstance(subfield, str):
                output[subfield] = value
    return output

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
    keys = list(mappings.keys())
    for n in range(len(keys)):
        key = keys[n]
        key = key.split(".")[-1]
        # key = ".".join([key[0], key[-1]])
        keys[n] = key
        
    return dict(zip(keys, list(mappings.values())))

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
    xsl_tree = etree.parse(xsl_file)

    # Create an XSLT transformer
    transform = etree.XSLT(xsl_tree)

    # Apply the transformation
    result_tree = transform(xml_tree)
    return str(result_tree)
