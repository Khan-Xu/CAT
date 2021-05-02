#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_source_utils: source file construction. 

functions: _create_multi_source_file - create a cache h5py file of source.
           _save_source_file         - save calcualte wavefront to a cache file.
           _reconstruct_source_file  - cosntruct root source file from the 
                                       cachle source file from every process.
                                       
classes  : none.
"""

#-----------------------------------------------------------------------------#
# library

import os
import numpy as np
import h5py  as h
import random
import scipy.sparse.linalg as ssl

from cat.utils import _multi
from cat.utils._srw_utils import _undulator
from cat.utils._srw_utils import _srw_electron_beam
from cat.utils._srw_utils import _propagate_wave_front

from cat.utils import _decomposition
from cat.utils import _support
from cat.utils import _file_utils

#-----------------------------------------------------------------------------#
# constant

_N_SVD_TOP = 5000
_N_SVD_OPT = 2500
_N_SVD_TOL = 1500
_CUT_OFF = 1000

#-----------------------------------------------------------------------------#
# function

def _multi_layer_svd_exceed_0(
    n_electron, xcount, ycount, k_cut_off, file_name = "test.h5"
    ):
    
    """
    ---------------------------------------------------------------------------
    description: perform coherent mode decompostion by multi layer svd methods.
                 self._exceed = 0 
    
    args: xcount    - the pixel number of screen (x).
          ycount    - the pixel number of screen (y).
          k_cut_off - the cut off index of coherent mode.
          file_name - file name of the saved source.

    return: none.
    ---------------------------------------------------------------------------
    """

    # use the scipy svds algorthm
    
    import scipy.sparse.linalg as ssl
    
    # multi-process parameters
    
    n_rank = _multi._get_size()
    c_rank = _multi._get_rank()
    
    # root process
    
    if c_rank == 0:
        
        pre_cmodes = np.zeros(
            (xcount * ycount, k_cut_off * (n_rank - 1)), 
            dtype = complex
            )
        flag = 0
        
        for i in range(1, n_rank):
            
            icore_cmode = _multi._recv(
                (xcount * ycount, k_cut_off), np_dtype = np.complex128, 
                dtype = _multi.c, source = i, tag = i
                )
            pre_cmodes[:, flag : flag + k_cut_off] = icore_cmode
            flag += k_cut_off
        
        # further calcuated the final svd results
        
        svd_matrix = pre_cmodes
        vector, value, evolution = ssl.svds(svd_matrix, k = int(2*k_cut_off))
        
        eig_vector = np.copy(vector[:, ::-1], order = 'C')
        value = np.copy(np.abs(value[::-1]), order = 'C')

        with h.File(file_name, 'a') as f:    
            
            coherence_dict = f.create_group("coherence")
            coherence_dict.create_dataset("eig_vector", data = eig_vector)
            coherence_dict.create_dataset("eig_value", data = value)
        
        
    elif c_rank > 0:
        
        crank_file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (c_rank)

        with h.File(crank_file_name, 'a') as crank_file:
            crank_wfrs = np.array(crank_file['wave_front/arex'])
        os.remove(crank_file_name)

        vectors, values, evolution = ssl.svds(
            crank_wfrs.T, k = int(2*k_cut_off)
            )
        crank_vectors = np.copy(vectors[:, ::-1], order = 'C')
        crank_value = np.copy(np.abs(values[::-1], order = 'C'))
        crank_vectors = crank_vectors * crank_value
        crank_vectors = np.array(crank_vectors[:, 0 : k_cut_off])

        _multi._send(
            crank_vectors, dtype = _multi.c, 
            dest = 0, tag = c_rank
            )

def _multi_layer_svd_exceed_1(
    n_electron, xcount, ycount, k_cut_off, file_name = "test.h5"
    ):
    
    """
    ---------------------------------------------------------------------------
    description: perform coherent mode decompostion by multi layer svd methods.
                 self._exceed = 1 
    
    args: xcount        - the pixel number of screen (x).
          ycount        - the pixel number of screen (y).
          k_cut_off     - the cut off index of coherent mode.
          file_name     - file name of the saved source.

    return: none.
    ---------------------------------------------------------------------------
    """

    # use the scipy svds algorthm
    
    import scipy.sparse.linalg as ssl
    
    # multi-process parameters
    
    n_rank = _multi._get_size()
    c_rank = _multi._get_rank()
    r_rank = int(n_electron / _N_SVD_OPT)

    # reset electrons

    if n_electron > _N_SVD_TOL * (n_rank - 1) and n_electron < _N_SVD_TOP * (n_rank - 1):
        pass
    else:
        n_electron = n_electron + (_N_SVD_OPT - n_electron % _N_SVD_OPT)

    # root process
    
    if c_rank == 0:
        
        wfr_arrays = np.zeros((n_electron, int(xcount * ycount)), dtype = complex)
        
        start_index = 0
        end_index = 0

        for i in range(1, n_rank):
            crank_file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (i)

            with h.File(crank_file_name, 'a') as crank_file:
                crank_wfrs = np.array(crank_file['wave_front/arex'])
                
                print(np.shape(crank_wfrs), flush = True)

                end_index += np.shape(crank_wfrs)[0]

                print(start_index, flush = True)
                print(end_index, flush = True)
                wfr_arrays[start_index : end_index, :] = crank_wfrs
                start_index += np.shape(crank_wfrs)[0]
 
            os.remove(crank_file_name)

        start_index = 0
        end_index = 0

        for i in range(1, r_rank + 1):

            end_index += _N_SVD_OPT
            _multi._send(wfr_arrays[start_index : end_index, :], dtype = _multi.c, dest = i, tag = i)
            start_index += _N_SVD_OPT

        pre_cmodes = np.zeros(
            (xcount * ycount, k_cut_off * r_rank), 
            dtype = complex
            )
        flag = 0
        
        for i in range(1, r_rank + 1):
            
            icore_cmode = _multi._recv(
                (xcount * ycount, k_cut_off), np_dtype = np.complex128, 
                dtype = _multi.c, source = i, tag = i
                )
            pre_cmodes[:, flag : flag + k_cut_off] = icore_cmode
            flag += k_cut_off
        
        # further calcuated the final svd results
        
        svd_matrix = pre_cmodes
        vector, value, evolution = ssl.svds(svd_matrix, k = int(2*k_cut_off))
        
        eig_vector = np.copy(vector[:, ::-1], order = 'C')
        value = np.copy(np.abs(value[::-1]), order = 'C')

        with h.File(file_name, 'a') as f:    
            
            coherence_dict = f.create_group("coherence")
            coherence_dict.create_dataset("eig_vector", data = eig_vector)
            coherence_dict.create_dataset("eig_value", data = value)
        
        
    elif c_rank > 0:
        
        if c_rank in list(range(1, r_rank + 1)):
            
            crank_wfrs = _multi._recv(
                (_N_SVD_OPT, int(xcount * ycount)), np_dtype = np.complex128, 
                dtype = _multi.c, source = 0, tag = c_rank
                )

            vectors, values, evolution = ssl.svds(
                crank_wfrs.T, k = int(2*k_cut_off)
                )
            crank_vectors = np.copy(vectors[:, ::-1], order = 'C')
            crank_value = np.copy(np.abs(values[::-1], order = 'C'))
            crank_vectors = crank_vectors * crank_value
            crank_vectors = np.array(crank_vectors[:, 0 : k_cut_off])

            print(np.shape(crank_vectors), flush = True)

            _multi._send(
                crank_vectors, dtype = _multi.c, 
                dest = 0, tag = c_rank
                )
        
        else:
            pass

def _multi_layer_svd_exceed_2(
    n_electron, xcount, ycount, k_cut_off, file_name = "test.h5"
    ):
    
    """
    ---------------------------------------------------------------------------
    description: perform coherent mode decompostion by multi layer svd methods.
                 self._exceed = 1 
    
    args: xcount        - the pixel number of screen (x).
          ycount        - the pixel number of screen (y).
          k_cut_off     - the cut off index of coherent mode.
          file_name     - file name of the saved source.

    return: none.
    ---------------------------------------------------------------------------
    """

    # use the scipy svds algorthm
    
    import scipy.sparse.linalg as ssl
    
    # multi-process parameters
    
    n_rank = _multi._get_size()
    c_rank = _multi._get_rank()
    r_rank = int(n_electron / _N_SVD_OPT)

    # root process
    
    if c_rank == 0:
        
        wfr_arrays = np.zeros((n_electron, int(xcount * ycount)), dtype = complex)
        
        start_index = 0
        end_index = 0

        for i in range(1, n_rank):
            crank_file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (i)

            with h.File(crank_file_name, 'a') as crank_file:
                crank_wfrs = np.array(crank_file['wave_front/arex'])
                
                end_index += np.shape(crank_wfrs)[0]
                wfr_arrays[start_index : end_index, :] = crank_wfrs
                start_index += np.shape(crank_wfrs)[0]
 
            os.remove(crank_file_name)

        start_index = 0
        end_index = 0

        for i in range(1, r_rank + 1):

            end_index += _N_SVD_OPT
            _multi._send(wfr_arrays[start_index : end_index, :], dtype = _multi.c, dest = i, tag = i)
            start_index += _N_SVD_OPT

        pre_cmodes = np.zeros(
            (xcount * ycount, k_cut_off * r_rank), 
            dtype = complex
            )
        flag = 0
        
        for i in range(1, r_rank + 1):
            
            icore_cmode = _multi._recv(
                (xcount * ycount, k_cut_off), np_dtype = np.complex128, 
                dtype = _multi.c, source = i, tag = i
                )
            pre_cmodes[:, flag : flag + k_cut_off] = icore_cmode
            flag += k_cut_off
        
        # further calcuated the final svd results
        
        svd_matrix = pre_cmodes
        vector, value, evolution = ssl.svds(svd_matrix, k = int(2*k_cut_off))
        
        eig_vector = np.copy(vector[:, ::-1], order = 'C')
        value = np.copy(np.abs(value[::-1]), order = 'C')

        if os.path.isfile(file_name): os.remove(file_name)

        with h.File(file_name, 'a') as f:    
            
            coherence_dict = f.create_group("coherence")
            coherence_dict.create_dataset("eig_vector", data = eig_vector)
            coherence_dict.create_dataset("eig_value", data = value)
        
        
    elif c_rank > 0:
        
        if c_rank in list(range(1, r_rank + 1)):
            pass
        else:
            crank_wfrs = _multi._recv(
                (_N_SVD_OPT, int(xcount * ycount)), np_dtype = np.complex128, 
                dtype = _multi.c, source = 0, tag = c_rank
                )

            vectors, values, evolution = ssl.svds(
                crank_wfrs.T, k = int(2*k_cut_off)
                )
            crank_vectors = np.copy(vectors[:, ::-1], order = 'C')
            crank_value = np.copy(np.abs(values[::-1], order = 'C'))
            crank_vectors = crank_vectors * crank_value
            crank_vectors = np.array(crank_vectors[:, 0 : k_cut_off])

            print(np.shape(crank_vectors), flush = True)

            _multi._send(
                crank_vectors, dtype = _multi.c, 
                dest = 0, tag = c_rank
                )

#-----------------------------------------------------------------------------#