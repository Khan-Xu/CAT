#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_file_utils: source file construction. 

functions: _create_multi_source_file - create a cache h5py file of source.
           _save_source_file         - save calcualte wavefront to a cache file.
           _reconstruct_source_file  - cosntruct root source file from the 
                                       cachle source file from every process.
                                       
classes  : none.
"""

#------------------------------------------------------------------------------
# library

import os
import numpy as np
import h5py  as h

from cat._source import _support
from cat._source import _multi

#------------------------------------------------------------------------------
# constant

#------------------------------------------------------------------------------
# function

#------------------------------------------------------------------------------
# create and save wavefronts in child process

def _create_multi_wfrs(c_rank, n_electron, n_points, file_name):
    
    """
    --------------------------------------------------------------------------
    description: create a cache h5py file of one process to save wavefronts.
    
    args: c_rank     - the cache file is created in rank > 0 processes.
          n_electron - the number of electron to put.
          n_points   - the pixel number of screen.
          file_name  - the filename of source. 
         
    return: none.
    --------------------------------------------------------------------------
    """

    # create cache file in rank > 0
    
    if c_rank > 0:
        file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (c_rank)
    
    file = _support._require_h5file(file_name)
    
    # create group to store the wave_fronts
        
    wave_front = file.create_group('wave_front')
    wfr_array = np.zeros((n_electron, n_points), dtype = 'complex')
    wave_front.create_dataset('arex', data = np.copy(wfr_array))
    wave_front.create_dataset('index', data = n_electron)

    file.close()

def _save_multi_wfrs(c_rank, wfr_array, start_index, end_index, file_name):

    """
    ---------------------------------------------------------------------------
    description: save calcualted wavefront to a cache file in one process.
    
    args: c_rank      - the cache file is created in rank > 0 processes.
          wfr_array   - the wavefronts to save.
          start_index - the start index of the wfrs.
          end_index   - the end index of the wfrs.
          file_name   - the filename of source. 
         
    return: none.
    ---------------------------------------------------------------------------
    """

    # open cache file in rank > 0
    
    c_file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (c_rank)

    with h.File(c_file_name, 'a') as cache_file:

        wave_front = cache_file['wave_front']
        wave_front['arex'][start_index : end_index, :] = wfr_array

#------------------------------------------------------------------------------
# create and save vectors in child process

def _create_multi_vectors(c_rank, n_vector, n_points, file_name):
    
    """
    --------------------------------------------------------------------------
    description: create a cache h5py file of one process to save wavefronts.
    
    args: c_rank     - the cache file is created in rank > 0 processes.
          n_electron - the number of electron to put.
          n_points   - the pixel number of screen.
          file_name  - the filename of source. 
         
    return: none.
    --------------------------------------------------------------------------
    """

    # create cache file in rank > 0
    
    if c_rank > 0:
        file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (c_rank)
    
    file = _support._require_h5file(file_name)
    
    # create group to store the wave_fronts
        
    vector = file.create_group("vector")
    vector_array = np.zeros((n_points, n_vector), dtype = 'complex')
    vector.create_dataset('eig_vector', data = np.copy(vector_array))
    vector.create_dataset('eig_value', data = np.zeros(n_vector))
    vector.create_dataset('index', data = n_vector)

    file.close()

def _save_multi_vectors(c_rank, vector_array, value, start_index, end_index, file_name):

    """
    ---------------------------------------------------------------------------
    description: save calcualted wavefront to a cache file in one process.
    
    args: c_rank       - the cache file is created in rank > 0 processes.
          vector_array - the vector_array to save.
          value        - the eig_value to save.
          start_index  - the start index of the wfrs.
          end_index    - the end index of the wfrs.
          file_name    - the filename of source. 
         
    return: none.
    ---------------------------------------------------------------------------
    """

    # open cache file in rank > 0
    
    c_file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (c_rank)

    with h.File(c_file_name, 'a') as cache_file:

        vector = cache_file['vector']
        vector['eig_vector'][:, start_index : end_index] = vector_array
        vector['eig_value'][start_index : end_index] = value

#------------------------------------------------------------------------------
# construct all wavefronts to one files

def _construct_cache_file(
        file_name, n_count, n_points, electron_beam, undulator, screen, wavelength, 
        n_rank, option = 'wave_front'
        ):
    
    """
    ---------------------------------------------------------------------------
    description: cosntruct root source file from the cachle source file from 
                 every process.
    
    args: file_name     - file name
          n_count       - the number of electron to put.
          n_points      - the pixel number of screen.
          electron_beam - parameters of e_beam.
          undulator     - parameters of undualtor.
          screen        - parameters of screen.
          n_rank        - total rank number.
          option        - wavefront or eig_vector

    return: none.
    ---------------------------------------------------------------------------
    """
    
    source_file = _support._require_h5file(file_name)
    
    # save parameters of e_beam, undualtor and screen.
    
    descript = source_file.create_group("description")
    _support._dict_to_h5(descript, electron_beam)
    _support._dict_to_h5(descript, undulator)
    _support._dict_to_h5(descript, screen)
    descript.create_dataset("wavelength", data = wavelength)
    
    # create wavefront group
    
    data_array = np.zeros((n_count, n_points), dtype = 'complex')
    data_to_save = np.copy(data_array)
    
    c_count = 0
    
    # loading data from every cache data
    
    for i in range(1, n_rank):
        
        i_file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (i)
        i_file = h.File(i_file_name, 'a')
        
        end = c_count + np.array(i_file[option + "/index"])
        data_to_save[c_count : end, :] = np.array(i_file[option + "/arex"])
        c_count = c_count + np.array(i_file[option + "index"])

        if os.path.isfile(i_file_name): os.remove(i_file_name)
   
    cache_file = _support._require_h5file("_cache.h5")
    data_array = cache_file.create_group(option)
    data_array.create_dataset("arex", data = data_to_save)
    cache_file.close()

def _construct_source_file(file_name, electron_beam, undulator, screen, wavelength):
    
    """
    ---------------------------------------------------------------------------
    description: cosntruct source files.
    
    args: file_name     - file name
          n_count       - the number of electron to put.
          n_points      - the pixel number of screen.
          electron_beam - parameters of e_beam.
          undulator     - parameters of undualtor.
          screen        - parameters of screen.
          n_rank        - total rank number.
          option        - wavefront or eig_vector

    return: none.
    ---------------------------------------------------------------------------
    """
    
    source_file = _support._require_h5file(file_name)
    
    # save parameters of e_beam, undualtor and screen.
    
    descript = source_file.create_group("description")
    _support._dict_to_h5(descript, electron_beam)
    _support._dict_to_h5(descript, undulator)
    _support._dict_to_h5(descript, screen)
    descript.create_dataset("wavelength", data = wavelength)
    
    source_file.close()

#-----------------------------------------------------------------------------#
# class

#-----------------------------------------------------------------------------#
