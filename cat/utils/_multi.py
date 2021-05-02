#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_multi : The multiprocess tools base on mpi4py

Functions: _send     - block communicate send
           _recv     - block communicate recv
           _get_rank - get the rank number of current process
           _get_size - get the size of the rank
           
Classes  : None 
"""

#-----------------------------------------------------------------------------#
# library

import mpi4py.MPI as mpi
import numpy      as np

#-----------------------------------------------------------------------------#
# constant

a = mpi.ANY_SOURCE  # recv from any source
i = mpi.INT         # int var
f = mpi.FLOAT       # float var
c = mpi.COMPLEX     # complex var

#-----------------------------------------------------------------------------#
# function

def _send(data, dtype = f, dest = 0, tag = 0): 
    
    """
    Block comunicate send.
    
    Args: data  - The data to send. numpy array is used
          dtype - The type of data, should be consist with data.
          dest  - The dest process of this sending.
          tag   - The tag of this sending. 
          
    Return: None.
    """
    
    # numpy array is used in this function
    
    if isinstance(data, np.ndarray):   
        mpi.COMM_WORLD.Send([data, dtype], dest = dest, tag = tag)
    else:
        raise ValueError("data is not numpy array")

        
def _recv(size, np_dtype = np.float, dtype = f, source = 0, 
          tag = 0, mode = 'auto'):
    
    """
    Block comunicate recv.
    
    Args: size   - The size of the data to recive.
          data   - The data to recv. numpy array is used.
          dtype  - The type of data, should be consist with data.
          source - The dest process of this sending.
          tag    - The tag of this reciving. 
          
    Return: recv_data - recviced data.
    """
    
    if isinstance(size, (list, tuple)):
        data = np.zeros(size, dtype = np_dtype)
    else:
        data = np.zeros(int(size), dtype = np_dtype)

    if isinstance(data, np.ndarray):
        mpi.COMM_WORLD.Recv([data, dtype], source = source, tag = tag)
    else:
        raise ValueError("data is not numpy array")
        
    return data

def _get_rank():
    
    """
    Get the rank number of current process.
    
    Args: None.
    
    Return: the rank number of current process.    
    """

    return mpi.COMM_WORLD.Get_rank()


def _get_size():
    
    """
    Get the process number.
    
    Args: None.
    
    Return: the process number.    
    """
    
    return mpi.COMM_WORLD.Get_size()


#-----------------------------------------------------------------------------#        

        
    