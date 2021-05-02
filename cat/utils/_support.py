#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_source_utils: Source construction support.

functions: _cal_part       - divide n_tot into seveal parts with the length 
                             n_divide.
           _cal_rank_part  - divide n_tot into (n_rank - 1) parts.
           _dict_to_h5     - save python dict to a h5py group with dataset.
           _require_h5file - create a h5py file, remove it and create a new one 
                             if exist.
           
classes  : none.
"""

#-----------------------------------------------------------------------------#
# library

import os
import numpy as np
import h5py  as h

from cat.utils import _multi

#-----------------------------------------------------------------------------#
# constant

_N_Part = 20 # the number of csd part.

#-----------------------------------------------------------------------------#
# function
    
def _cal_part(n_tot, n_divide):
    
    """
    ---------------------------------------------------------------------------
    description: divide n_tot into seveal parts with the length n_divide.
    
    for example: 
        n_tot    - 11 
        n_divide - 3 
        result   - index [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
                 - count [3, 3, 3, 2]
                 
    
    args: n_tot    - total number to divide.
          n_divide - the length of the part.

    return: count - a list contains the length of every part.
            index - a list contains the index  of every part.
    ---------------------------------------------------------------------------
    """
    
    n_tot = round(n_tot)
    n_divide = int(np.ceil(n_divide))
    index = list()
    count = list()
    
    if n_tot > n_divide:
        
        if n_tot % n_divide:
            
            n_part = int(n_tot // n_divide)
            n_rest = int(n_tot - n_divide * (n_tot // n_divide))
            
            for i in range(n_part):
                index.append(np.arange(i*n_divide, (i + 1)*n_divide))
                count.append(n_divide)
                
            index.append(np.arange(n_tot)[(i + 1)*n_divide :])
            count.append(n_rest)
   
        else:
            
            n_part = int(n_tot // n_divide)
            n_rest = 0
            
            for i in range(n_part):
                index.append(np.arange(i*n_divide, (i + 1)*n_divide))
                count.append(n_divide)
    
    else:
        
        index.append(np.arange(0, n_tot))
        count.append(n_tot)
    
    return count, index

def _cal_rank_part(n_tot, n_rank):
    
    """
    ---------------------------------------------------------------------------
    description : 
        divide n_tot into (n_rank - 1) parts. 
        used to plan the distribution of multiprocess. 
        usullay, rank == 0 is not used for the distribution plan.
        
    for example: 
        n_tot  - 11
        n_rank - 4
        result - index [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
               - count [3, 3, 3, 2] 
               
    
    args: n_tot  - total number to divide.
          n_rank - the total rank number.

    return: count - a list contains the length of every part.  
            index - a list contains the index of every part.
    ---------------------------------------------------------------------------
    """
    
    n_tot = round(n_tot)
    n_rank = n_rank - 1
    index = list()
    count = list()
    
    if n_rank >= 2:
        
        if n_tot % n_rank:
            
            n_per_process = int(n_tot // n_rank)
            n_rest = int(n_tot - n_rank * (n_tot // n_rank))
            
            for i in range(n_rank - 1):
                index.append(np.arange(i*n_per_process, (i + 1)*n_per_process))
                count.append(n_per_process)
                
            index.append(np.arange(n_tot)[(i + 1)*n_per_process :])
            count.append(n_per_process + n_rest)
        
        else:
            
            n_per_process = int(n_tot // n_rank)
            n_rest = 0
            
            for i in range(n_rank):
                index.append(np.arange(i*n_per_process, (i + 1)*n_per_process))
                count.append(n_per_process)
    
    else:
        
        index.append(np.arange(0, n_tot))
        count.append(n_tot)
    
    return count, index

def _dict_to_h5(group, dict_file):
    
    """
    ---------------------------------------------------------------------------
    description: save python dict to a h5py group with dataset.
    
    args: group     - h5py group.
          dict_file - python dict to save.
         
    return: none.
    ---------------------------------------------------------------------------
    """
    
    for key, value in dict_file.items():
        group.create_dataset(key, data = value)
        
def _require_h5file(file_name):
    
    """
    ---------------------------------------------------------------------------
    description: create a h5py file, remove it and create a new one if exist.
    
    args: file_name - the name of the h5py file.
         
    return: h5file handle.
    ---------------------------------------------------------------------------
    """
    
    if os.path.isfile(file_name): os.remove(file_name)
    
    return h.File(file_name, 'a')

def _shift_plane(offset, tick, start, end, count):

    """
    ---------------------------------------------------------------------------
    description: shift a segment.
    
    args: offset - shift value.
          tick   - the coordinate of the range.
          start  - tick[0].
          end    - tick[-1].
          count  - len(tick).
         
    return: .
    ---------------------------------------------------------------------------
    """

    def _locate(xtick, value):

            # locate the range of shift wfr

            if value > np.max(xtick) or value < np.min(xtick):
                raise ValueError("the given value is out of range")
            else:
                return np.argmin(np.abs(xtick - value))

    if offset > 0:

        locl0 = _locate(tick, start + offset)
        locr0 = count - locl0
        locl1 = 0
        locr1 = count - 2*locl0

    elif offset < 0:

        locr0 = _locate(tick, end + offset)
        locl0 = count - locr0
        locl1 = count - 2*locr0
        locr1 = count

    return locl0, locr0, locl1, locr1 

