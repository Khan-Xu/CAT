#-----------------------------------------------------------------------------#
# Copyright (c) 2021 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS HXS (B4) xuhan@ihep.ac.cn"
__date__     = "Date : 04.01.2021"
__version__  = "beta-1.0"


"""
source: The construction of coherent modes or wavefronts.

Functions: _create_source_propagte - calcualte wavefronts from source to 
                                     screen.
           multi_electron          - calculate coherent modes.
           single_electron         - calcualte a wavefront from single 
                                     electron.
           
Classes  : None.
"""

#-----------------------------------------------------------------------------#
# library

import random
import numpy as np

from cat.utils import _multi
from cat.utils import _source_utils

from cat.utils._srw_utils import _undulator
from cat.utils._srw_utils import _srw_electron_beam
from cat.utils._srw_utils import _propagate_wave_front

#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function
    
def _create_source_propagate(undulator, electron_beam, screen, file):
    
    """
    Calculate wavefronts from electron source to screens.
    
    Args: undulator     - the parameters of undualtor.
          electron_beam - the parameters of electron_beam.
          screen        - the parameters of screen.
          file          - file name.
    
    Return: None.
    """
    
    # The multi_process parameters
    
    rank = _multi._get_rank()
    n_rank = _multi._get_size()
    
    # The construction of undulator
    
    und = _undulator(undulator)
    und.wave_length()
    und.cal_k(electron_beam_energy = electron_beam["energy"])
    und_magnetic_structure = und.magnetic_structure()
    
    # The construction of electron beam
    
    e_beam = _srw_electron_beam(electron_beam, und.n_period, und.period_length)
    e_beam.monte_carlo()
    
    # The parameters of undulator and beam
    
    n_electron = electron_beam['n_electron']
    n_points = screen['nx'] * screen['ny']
    elec_count, elec_index = _source_utils._cal_rank_part(n_electron, n_rank)
    
    # Set monte carlo random seed
    
    random.seed(rank * 123)
    newSeed = random.randint(0, 1000000)
    random.seed(newSeed)

    #------------------------------------------------
    # wavefront calculation
    
    if rank > 0:
        
        n_electron_per_process = elec_count[rank - 1]
        _source_utils._create_multi_source_file(
            rank, n_electron_per_process, n_points, file_name = file
            )
        
        for i in range(n_electron_per_process):
            
            # random parameters
            
            rand_array = [random.gauss(0, 1) for ir in range(5)]
            
            # monte carlo process
            
            e_beam.monte_carlo()
            part_beam, wavelength, resonance_energy = e_beam.after_monte_carlo(
                rand_array, und.period_length, und.k_vertical, und.k_horizontal
                )
            
            # The magnetic structure setting  
            
            wavefront = _propagate_wave_front(screen, resonance_energy)
            wavefront._cal_wave_front(part_beam, und_magnetic_structure)
            
            # send wavefront
            
            _source_utils._save_source_file(
                i, rank, n_points, file, part_beam, wavefront, screen['nx'], 
                screen['ny'], resonance_energy
                )
            
            if rank == 1 and (10*i) % (10*int(n_electron_per_process/10)) == 0:
                print("| %.2d percent                      |" 
                      % (int(100*i/n_electron_per_process)), 
                      flush = True)
        
        # source_file.close()
                
        _multi._send(np.array(1, dtype = 'i'), dtype = _multi.i, tag = rank)
        
        #------------------------------------------------ 
        
    if rank == 0:
        
        print("___________________________________", flush = True)
        print("| wavefront calculation start.    |", flush = True)
        
        for i in range(n_rank - 1):
            flag = _multi._recv(1, 
                                np_dtype = np.int, 
                                dtype = _multi.i, 
                                source = i + 1, 
                                tag = i + 1)
            
        _source_utils._reconstruct_source_file(
            file, n_electron, n_points, electron_beam, undulator, screen, 
            und.wavelength, n_rank
            )
        
        print("| wavefront calculation finished. |", flush = True)
        print("___________________________________", flush = True)
            
        _source_utils._cal_csd(n_electron, 
                               n_points, 
                               screen["nx"], 
                               screen["ny"], 
                               file_name = file)
        # _source_utils._scipy_sparse_cmd(screen["n_vector"], file_name = file)

#-----------------------------------------------------------------------------
def multi_electron(undulator, electron_beam, screen, file):
    
    """
    Calculate wavefront from multi-electron source to screens.
    
    Args: undulator     - the parameters of undualtor.
          electron_beam - the parameters of electron_beam.
          screen        - the parameters of screen.
          file          - file name.
    
    Return: None.
    """
    
    rank = _multi._get_rank()
    
    _create_source_propagate(undulator, electron_beam, screen, file)
    
    # Choose different CMD methods for different size matrix. Depend on the 
    # experience, 150**2 is setted as a boundary.
    
    if screen["nx"]*screen["ny"] < 150**2:
        if rank == 0:
            
            # scipy.sparse.linalg.eigsh is arnoldi method, single process
            
            _source_utils._scipy_sparse_cmd(n_vector = screen["n_vector"], 
                                            file_name = file)
    else:
        
        # if matrix is too large, multi-process arnoldi method is used.
        
        _source_utils._arnoldi_cmd(screen["nx"]*screen["ny"],
                                   n_vector = screen["n_vector"],
                                   file_name = file)

#------------------------------------------------------------------------------    
def single_electron(undulator, electron_beam, screen):
    
    """
    Calculate wavefront from single lectron source to screens.
    
    Args: undulator     - the parameters of undualtor.
          electron_beam - the parameters of electron_beam.
          screen        - the parameters of screen.

    Return: electron - dict contains wavefront, intensity, sumed intensity
                       along axis x and y.
    """
    
    # The construction of undulator
    
    und = _undulator(undulator)
    und.wave_length()
    und.cal_k(electron_beam_energy = electron_beam["energy"])
    und_magnetic_structure = und.magnetic_structure()
    
    # The construction of electron beam
    
    e_beam = _srw_electron_beam(electron_beam, und.n_period, und.period_length)
    e_beam.monte_carlo()
    part_beam, wavelength, resonance_energy = e_beam.after_monte_carlo(
                [0, 0, 0, 0, 0], und.period_length, und.k_vertical, 
                und.k_horizontal
                )
    
    # The magnetic structure setting 
    
    wavefront = _propagate_wave_front(screen, resonance_energy)
    wavefront._cal_wave_front(part_beam, und_magnetic_structure)
    
    # Saving wavefront result.
    
    # The mesh of result.
    
    x = np.linspace(wavefront.wfr.mesh.xStart, wavefront.wfr.mesh.xFin, 
                    wavefront.wfr.mesh.nx)
    y = np.linspace(wavefront.wfr.mesh.yStart, wavefront.wfr.mesh.yFin, 
                    wavefront.wfr.mesh.ny)
    
    # The wavefront.
    
    wfr0 = np.reshape(
        wavefront.wfr.arEx, 
        [wavefront.wfr.mesh.nx, wavefront.wfr.mesh.ny, 2]
        )
    
    # The intensity of wavefront.
    
    i0 = np.sum(np.abs(wfr0)**2, 2)
    
    electron = {'wfr': wfr0, 'intensity': i0,
                'ix': np.sum(i0, axis = 0), 'iy': np.sum(i0, axis = 1),
                'x': x, 'y': y}
    
    return electron

#-----------------------------------------------------------------------------#
