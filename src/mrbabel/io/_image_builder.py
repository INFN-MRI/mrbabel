"""Build images from raw data header."""

__all__ = ["ImageBuilder"]

import copy
import os

import numpy as np
import pydicom
import multiprocessing

from multiprocessing.dummy import Pool as ThreadPool

import mrd
from mrd import ImageHeader, ImageArray, Image

from ..utils import get_geometry

from .converters._mrd2dicom import dump_dicom_images
from .sorting import unsort_images

class ImageBuilder:
    """Build images from raw data header."""
    
    def __init__(self, head, acq_head):
        _image_head = _raw2head(head, acq_head.ravel()[0])
        
        # get geometry
        geometry = get_geometry(head, acq_head)
        
        # build slice locations
        nslices = int(geometry.shape[-1])
        spacing = geometry.fov[-1] / nslices
        vector = np.asarray(geometry.normal)
        
        # get initial position
        center = np.asarray(acq_head.ravel()[0].position)
        pos0 = center - (nslices // 2) * vector
        
        # get positions for each slice
        position = [pos0 + n * spacing * vector for n in range(nslices)]
        
        # build template
        headers = []
        for n in range(nslices):
            _tmp = copy.deepcopy(_image_head)
            _tmp.position = position[n]
            _tmp.slice = n
            headers.append(_tmp)
            
        self.image_template = ImageArray(headers=np.asarray(headers))
        self.head = head
        self.spacing = spacing
        
    def write_dicom(
            self,
            series_description,
            img,
            outpath,
            series_number_scale = 1000,
            series_number_offset = 0,
            ):
        img = copy.deepcopy(img)
        
        # cast image
        minval = np.iinfo(np.int16).min
        maxval = np.iinfo(np.int16).max
        img[img < minval] = minval
        img[img > maxval] = maxval
        img = img.astype(np.int16)
        
        # build image from input and template
        mrd_image = copy.deepcopy(self.image_template)
        mrd_image.data = img
        
        # dump to dicom
        mrd_images = [Image(data=mrd_image.data[n], head=mrd_image.headers[n]) for n in range(img.shape[0])]
        dsets = dump_dicom_images(mrd_images, self.head)
        
        # adjust dicom
        SeriesInstanceUID = pydicom.uid.generate_uid()
                    
        # count number of instances
        ninstances = img.shape[0]
                
        # generate series number
        series_number = str(series_number_scale * int(dsets[0].SeriesNumber) + series_number_offset)
        
        # get level
        windowMin = np.percentile(img, 5)
        windowMax = np.percentile(img, 95)                   
        windowWidth = windowMax - windowMin
            
        # set properties
        for n in range(ninstances):     
            dsets[n].SliceThickness = str(self.spacing)
            # dsets[n].SliceThickness = str(spacing)
            dsets[n].WindowWidth = str(windowWidth)
            dsets[n].WindowCenter = str(0.5 * windowWidth)
    
            dsets[n].SeriesDescription = series_description
            dsets[n].SeriesNumber = series_number
            dsets[n].SeriesInstanceUID = SeriesInstanceUID
        
            dsets[n].SOPInstanceUID = pydicom.uid.generate_uid()
            dsets[n].InstanceNumber = str(n + 1)
            
            try:
                dsets[n].ImagesInAcquisition = ninstances
            except:
                pass
            try:
                dsets[n][0x0025, 0x1007].value = ninstances
            except:
                pass
            try:
                dsets[n][0x0025, 0x1019].value = ninstances
            except:
                pass        
        
        # save dicom
        # Create destination folder
        outpath = os.path.realpath(outpath)
        if os.path.exists(outpath) is False:
            os.makedirs(outpath)
        paths = [
            os.path.join(outpath, f"image-{str(n).zfill(4)}.dcm") for n in range(len(dsets))
        ]
        
        def dcmwrite(filename, dataset):
            pydicom.dcmwrite(
                filename,
                dataset,
                enforce_file_format=True,
                little_endian=True,
                implicit_vr=False,
            )
        
        # Writing
        paths_dsets = [(paths[n], dsets[n]) for n in range(len(dsets))]
        with ThreadPool(multiprocessing.cpu_count()) as pool:
            pool.starmap(dcmwrite, paths_dsets)
        
    def write_nifti(
            self,
            series_description,
            img,
            outpath,
            series_number_scale = 1000,
            series_number_offset = 0,
            ):
        raise NotImplemented
        
# local utils
def _raw2head(head, acq_head):
    """Populate ImageHeader fields from AcquisitionHeader"""
    img_head = ImageHeader(image_type=mrd.ImageType.MAGNITUDE)

    img_head.field_of_view = np.asarray((head.encoding[-1].encoded_space.field_of_view_mm.x,
                                         head.encoding[-1].encoded_space.field_of_view_mm.y,
                                         head.encoding[-1].encoded_space.field_of_view_mm.z))

    img_head.flags  = acq_head.flags
    img_head.measurement_uid = acq_head.measurement_uid

    img_head.position = acq_head.position
    img_head.line_dir = acq_head.read_dir
    img_head.col_dir = acq_head.phase_dir
    img_head.slice_dir = acq_head.slice_dir
    img_head.patient_table_position = acq_head.patient_table_position

    img_head.average = 0
    img_head.slice = 0
    img_head.contrast = 0
    img_head.phase = 0
    img_head.repetition = 0
    img_head.set = 0

    img_head.acquisition_time_stamp = acq_head.acquisition_time_stamp
    img_head.physiology_time_stamp = acq_head.physiology_time_stamp

    # Defaults, to be updated by the user
    img_head.image_index = 1
    img_head.image_series_index = 0

    img_head.user_float = acq_head.user_float
    img_head.user_int = acq_head.user_int

    return img_head