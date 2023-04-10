############################################################################
# BDV2IMS - Convert BigStitcher/BigDataViewer formatted images to Imaris IMS
# 
# Author: Mia Panlilio
# 2023-03-13
#
# Supported file formats
#   - HDF5 (recommended): .hdf5 and .ims
#   - Anything readable by python-bioformats, however only .tiff has been tested
#
# For conversion from TIFF, the current parsing structure assumes that there is 
# one TIFF per time point, per channel all in the same parent directory.
# 
# Main dependencies
#   - h5py: https://github.com/h5py/h5py 
#   - PyImarisWriter: https://github.com/imaris/ImarisWriter
#   - python-bioformats (if not converting from HDF5): https://github.com/CellProfiler/python-bioformats
#
# Development notes
#   + 2023-04-10: added python-bioformats reader
# 
# TODOS
#   - enable mutlichannel .tiff reading
#   - add .zarr support
#   - add .n5 support
#
############################################################################

import sys
import logging
sys.path.append("/ImarisWriter/python/")
from PyImarisWriter import PyImarisWriter as PW
from datetime import datetime
import time
import h5py
import numpy as np
import re
import os
import xml.etree.ElementTree as ET
import re
import multiprocessing
import argparse
import javabridge
import bioformats

logfmt = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(format=logfmt)

class converter:
    ###################################################
    # Constructor
    ###################################################
    def __init__(self, h5filename=None):
        self.h5filename = h5filename
        self.h5format = ''
        self.xmlfilename = ''
        self.datasets = {}
        self.config = {}

        self.logger = logging.getLogger('bdv_converter')
        self.logger.setLevel(15)

        self._unit_default = 'um'
        self._chunksize_default = [512,512,128]
        self._re_expr_channel = [r'channel(\d+)',r'c(\d+)',r'ch_(\d+)',r'ch(\d+)',r'C(\d+)']
        self._re_expr_timepoint = [r't(\d+)']
        self._re_unique_channel = True
        self._re_unique_timepoint = False

        self._sleep_time_init = 10
        self._sleep_time_copy = 3

        xmlfilename = f'{self.h5filename.split(".h5")[0]}.xml'
        self.reader = h5py.File
        if re.search('.ims$',self.h5filename):
            self.h5format = 'ims'
        elif os.path.exists(xmlfilename):
            self.h5format = 'bdv'
            self.xmlfilename = xmlfilename
        elif re.search('.h5$',self.h5filename):
            self.h5format = 'h5'
        else:
            self.reader = bf_like_h5
            self.h5format = 'bf'
            javabridge.start_vm(class_path=bioformats.JARS)

        self._index_datasets()

        vox_size, vox_unit = self.get_voxel_size()
        self.config['vox_size'] = vox_size
        self.config['vox_unit'] = vox_unit
        self.config['nchannels'] = 1+max([d['channel'] for _,d in self.datasets.items()])
        self.config['ntimepoints'] = 1+max([d['timepoint'] for _,d in self.datasets.items()])      

        imsize, chunksize, dtype = self.get_imsize()
        self.config['imsize'] = imsize
        self.config['dtype'] = dtype
        self.config['chunksize'] = chunksize

        self.config['extents'] = list(np.array(self.config['imsize'])*np.array(self.config['vox_size']))
        self.config['compression_algorithm'] = PW.eCompressionAlgorithmGzipLevel2
        self.config['adjust_color_range'] = False
        self.config['info'] = 'Converted using bdv2ims'
        self.config['threads'] = 16
        self.config['app_name'] = 'PyImarisWriter'
        self.config['app_ver'] = '1.0.0'

    ###################################################
    # MAIN CONVERSION
    ###################################################

    def run(self, destdir=None, threads = 16, chunksize=None, datatype=None, 
        compression_algorithm=PW.eCompressionAlgorithmGzipLevel2):
        
        destdir = destdir if destdir is not None else os.path.dirname(self.h5filename)
        imsname = os.path.join(destdir,'{}.ims'.format(os.path.basename(self.h5filename).split('.')[0]))

        self.config['chunksize'] = chunksize if chunksize is not None else self.config['chunksize']
        self.config['threads'] = threads if threads is not None else self.config['threads']
        self.config['dtype'] = datatype if datatype is not None else self.config['dtype']
        self.config['compression_algorithm'] = compression_algorithm if compression_algorithm is not None else self.config['compression_algorithm']
        self._check_dtype()
        self._check_chunksize()
        
        pw_imsize = PW.ImageSize(x=self.config['imsize'][0], y=self.config['imsize'][1], z=self.config['imsize'][2], c=self.config['nchannels'], t=self.config['ntimepoints'])
        pw_dimseq = PW.DimensionSequence('x','y','z','c','t')
        pw_chunksize = PW.ImageSize(x=self.config['chunksize'][0], y=self.config['chunksize'][1], z=self.config['chunksize'][2], c=1, t=1)
        pw_samplesize = PW.ImageSize(x=1,y=1,z=1,c=1,t=1)

        options = PW.Options()
        options.mNumberOfThreads = self.config['threads']
        options.mCompressionAlgorithmType = self.config['compression_algorithm']
        options.mEnableLogProgress = True

        pw_callback = PWCallback()

        converter = PW.ImageConverter(self.config['dtype'], pw_imsize, pw_samplesize, pw_dimseq, pw_chunksize, imsname, options, self.config['app_name'], self.config['app_ver'], pw_callback)

        time.sleep(self._sleep_time_init)

        self.logger.info('PyImarisWriter ImageConverter object created.')
        
        with self.reader(self.h5filename,'r') as f:
            block_index = PW.ImageSize()
            for dset,idx_dict in self.datasets.items():
                block_index.t = idx_dict['timepoint']
                block_index.c = idx_dict['channel']
                for z_idx,z in enumerate(range(0,pw_imsize.z,pw_chunksize.z)):
                    block_index.z = z_idx
                    zmax = min([pw_imsize.z,z+pw_chunksize.z])
                    for y_idx,y in enumerate(range(0,pw_imsize.y,pw_chunksize.y)):
                        block_index.y = y_idx
                        ymax = min([pw_imsize.y,y+pw_chunksize.y])
                        for x_idx,x in enumerate(range(0,pw_imsize.x,pw_chunksize.x)):
                            block_index.x = x_idx
                            xmax = min([pw_imsize.x,x+pw_chunksize.x])
                            if converter.NeedCopyBlock(block_index):
                                block = self.get_block(f[dset],x,y,z,xmax,ymax,zmax)
                                sz = block.shape
                                block = np.pad(block,((0,0),(0,pw_chunksize.y-sz[1]),(0,pw_chunksize.x-sz[2])))
                                self.logger.debug(f'Copying block at index {block_index}')
                                converter.CopyBlock(block,block_index)
                                time.sleep(self._sleep_time_copy)

        pw_extents = PW.ImageExtents(0,0,0,self.config['extents'][0],self.config['extents'][1],self.config['extents'][2])

        parameters = PW.Parameters()
        parameters.set_value('Image','Info','converted to IMS using PyImarisWriter')
        parameters.set_value('Image','Unit', self.config['vox_unit'])
        pw_timeinfo = [datetime.today()]
        pw_colorinfo = [PW.ColorInfo() for _ in range(pw_imsize.c)]

        converter.Finish(pw_extents,parameters,pw_timeinfo,pw_colorinfo,self.config['adjust_color_range'])
        converter.Destroy()

        self.logger.info('DONE! IMS was finalized and the converter has been closed.')

    
    ###################################################
    # Data retrieval 
    ###################################################    
    def get_block(self,file_handle,x,y,z,xmax,ymax,zmax):
        nx = xmax - x
        ny = ymax - y
        nz = zmax - z
        I = np.ndarray((nz,ny,nx),dtype=self.config['dtype'])
        if self.h5format=='bf':
            for idz,z in enumerate(range(z,zmax)):
                I[idz,:,:] = file_handle.read(z=z,rescale=False,XYWH=(x,y,nx,ny))
        else:
            I = np.array(file_handle[z:zmax,y:ymaz,x:xmax])
        return I

    ###################################################
    # Parameter retrieval
    ###################################################    
    
    def get_imsize(self):
        imsize, chunksize, dtype = getattr(self,f'_{self.h5format}_get_imsize')()
        return imsize, chunksize, dtype
    
    def _h5_get_imsize(self):
        with h5py.File(self.h5filename) as f:
            dset = next(iter(self.datasets))
            imsize =  f[dset].shape[-1::-1]
            dtype = str(f[dset].dtype)
            chunksize = f[dset].chunks[-1::-1]
        return imsize, chunksize, dtype

    def _bdv_get_imsize(self):
        return getattr(self,'_h5_get_imsize')()

    def _ims_get_imsize(self):
        return getattr(self,'_h5_get_imsize')()

    def _bf_get_imsize(self):
        xml = bioformats.get_omexml_metadata(path=self.h5filename)
        ome = bioformats.OMEXML(xml)
        imsize = [ome.image().Pixels.SizeX, ome.image().Pixels.SizeY, ome.image().Pixels.SizeZ]
        chunksize = self._chunksize_default
        dtype = ome.image().Pixels.PixelType
        return imsize, chunksize, dtype

    def _check_dtype(self):
        if self.config['dtype'][:3]=='int':
            self.logger.warning('Original datatype was a signed integer. Data will be converted to unsigned integer.')
            dtype = 'u'+self.config['dtype']
            self.config['dtype'] = dtype

    def _check_chunksize(self):
        if np.any(self.config['chunksize'] > self.config['imsize']):
            self.logger.warning(f'Chunksize of {self.config["chunksize"]} is larger than the image volume along one dimension. Setting chunk size to minima.')
            chunksize = np.minimum(self.config['chunksize'],self.config['imsize'])
            self.config['chunksize'] = chunksize

    def get_voxel_size(self):
        vx, vu = getattr(self,f'_{self.h5format}_get_voxel_size')()
        if len(vx)==0 or len(vu)==0:
            self.logger.warning('Voxel size and units could not be parsed correctly. You may set voxel_size and voxel_unit values manually in the config dictionary attribute. Otherwise the default 1x1x1 um^3 is used.')

            vx = [1,1,1]
            vu = self._unit_default
        return vx, vu

    def _bdv_get_voxel_size(self):
        vox_units = []
        vox_size = []
        tree = ET.parse(self.xmlfilename)
        root = tree.getroot()
        try:
            for view in root.findall('SequenceDescription/ViewSetups/ViewSetup/voxelSize'):
                vox_units = view[0].text
                vox_size = np.array(view[1].text.split(" ")).astype(float)
                break
            assert(len(vox_size)>0)
        except:
            self.logger.warning(f'No voxel size could be parsed from the detected XML at {self.xmlfilename}')    

        return vox_size, vox_units


    def _h5_get_voxel_size(self):
        vs = []
        vu = []
        try:
            with h5py.File(self.h5filename,'r') as f:
                vs,vu = f.visititems(self._h5_visit_res)
        except:
            pass
        return vs, vu


    def _ims_get_voxel_size(self):
        vs = []
        vu = []
        with h5py.File(self.h5filename,'r') as f:
            vs,vu = f.visititems(self._ims_visit_res)
        return vs,vu


    def _bf_get_voxel_size(self):
        xml = bioformats.get_omexml_metadata(path=self.h5filename)
        ome = bioformats.OMEXML(xml)
        vs = [ome.image().Pixels.PhysicalSizeX, ome.image().Pixels.PhysicalSizeY, ome.image().Pixels.PhysicalSizeZ]
        vu = ome.image().Pixels.PhysicalSizeXUnit
        return vs,vu

    def _index_datasets(self):
        dsets = self.list_datasets()
        counter_c = 1 if self._re_unique_channel else 0
        counter_t = 1 if self._re_unique_timepoint else 0
        for nd,d in enumerate(dsets):
            ch = self._index_parser(self._re_expr_channel,d)
            t = self._index_parser(self._re_expr_timepoint,d)
            self.datasets[d] = {'channel': ch if ch is not None else nd*counter_c,
                                'timepoint': t if t is not None else nd*counter_t}    
    
    ###################################################
    # HDF5 visiting functions and attribute parsing
    ###################################################

    def _h5_visit_res(self,name,h5obj):
        if isinstance(h5obj,h5py.Group) and re.search('resolutions',name):
            return h5obj[0], self._unit_default
        
    def _ims_visit_res(self,name,h5obj):
        if isinstance(h5obj,h5py.Group) and re.search('DataSetInfo/Image',name):
            vu = self._ims_attr2str(h5obj.attrs.get('Unit'))
            vs = []
            dims = ['X','Y','Z']
            for i,d in zip(range(3),dims):
                extmin = self._ims_attr2float(h5obj.attrs.get(f'ExtMin{i}'))
                extmax = self._ims_attr2float(h5obj.attrs.get(f'ExtMax{i}'))
                nvox = self._ims_attr2float(h5obj.attrs.get(f'{d}'))
                vs.append((extmax-extmin)/nvox)
            return vs,vu

    @staticmethod
    def _ims_attr2str(attr_array):
        s = list(attr_array.astype(str))
        s = ''.join(s)
        return s

    @classmethod
    def _ims_attr2float(self,attr_array):
        s = self._ims_attr2str(attr_array)
        return float(s)

    def list_datasets(self):
        if not self.h5format=='bf':
            h5ls = H5ls(highest_res_only=True)
            with h5py.File(self.h5filename,'r') as f:
                f.visititems(h5ls)
            names = h5ls.names
        else:
            _,ext = os.path.splitext(self.h5filename)
            parentdir = os.path.dirname(os.path.abspath(self.h5filename))
            names = []
            for f in os.listdir(parentdir):
                if f.endswith(ext):
                    names.append(f)
        return names

    @staticmethod
    def _index_parser(expr,dsetname):
        if not isinstance(expr,(list,tuple)):
            expr = [expr]
        found = False
        idx = None
        for p in expr:
            try:
                idx = int(re.findall(p,dsetname)[0])    
                found = True
            except:
                pass
            if found: break
        return idx

#######################################################################################################

class H5ls:
    """
    List all datasets in an HDF5 file, by default retrieving only those at the highest resolution. 
    Based on solution from 
    https://stackoverflow.com/questions/31146036/how-do-i-traverse-a-hdf5-file-using-h5py 
    user Bremsstrahlung
    """
    def __init__(self,highest_res_only=True):
        self.names = []
        self.highest_res_only = highest_res_only
        self.max_res = 0 

    def __call__(self,name,h5obj):
        if isinstance(h5obj,h5py.Dataset) and not name in self.names:
            sz = np.prod(h5obj.shape)
            if self.highest_res_only and sz > self.max_res:
                self.names = [name]
                self.max_res = sz
            elif self.highest_res_only and sz==self.max_res:
                self.names += [name]
            elif not self.highest_res_only:
                self.names += [name]


class PWCallback(PW.CallbackClass):
    def __init__(self):
        self.mUserDataProgress = 0
        self.logger = logging.getLogger('bdv_converter')
        
    def RecordProgress(self, progress, bytes_written):
        prg100 = int(progress * 100)
        if prg100 - self.mUserDataProgress >= 1:
            self.mUserDataProgress = prg100
            self.logger.info(f'Writing chunks...{prg100}% done. {bytes_written} bytes written')

########################################################################################################

class bf_like_h5:
    """
    Wraps bioformats reader objects to give it an h5py-like access pattern.
    """
    def __init__(self,dset,dummyflag):
        self.parent = os.path.dirname(os.path.abspath(dset))
        self.readers = {}

    def __enter__(self):
        return self

    def __getitem__(self,dset):
        if dset not in self.readers:
            self.readers[dset] = bioformats.ImageReader(os.path.join(self.parent,dset))
        return self.readers[dset]

    def __exit__(self,*args):
        for dset in self.readers:
            self.readers[dset].close()


########################################################################################################
def test():
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='bdv2ims',description='Convert HDF5 from the BigDataViewer/BigStitcher standard to Imaris')
    parser.add_argument('filename',type=str,help='File to be converted to IMS')
    parser.add_argument('destdir',default=None,type=str,nargs='?',help='Directory where IMS will be written. Defaults to the same directory as the source file.')
    parser.add_argument('-n','--nthreads',default=int(multiprocessing.cpu_count()-8),type=int,help='Number of threads for ImarisWriter to use. Defaults to (num detected CPUs) - 8. Some threads should be left free for data retrieval from the HDF5.')
    parser.add_argument('-d','--dtype',default=None,type=str,help='Numeric data type in which IMS should be written.')
    parser.add_argument('-c','--chunksize',default=None,type=int,nargs=3,help='Chunk size to use.')
    parser.add_argument('-a','--compression_algorithm',default=2,type=int,help='Compression algorithm to use, specified by an integer: no compression (0), gzip level N (1-9), shuffle gzip level N (11-19), lz4 (21), shuffle lz4 (31)')
    args = parser.parse_args()

    my_converter = converter(h5filename=args.filename)
    my_converter.run(destdir=args.destdir, threads=args.nthreads, chunksize=args.chunksize, datatype=args.dtype, compression_algorithm=args.compression_algorithm)

    if my_converter.h5format=='bf':
        javabridge.kill_vm()

