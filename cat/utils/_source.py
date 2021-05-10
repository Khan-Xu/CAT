#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
source: The construction of beam source.

Functions: create_source_propagte - calcualte wavefronts from source to screen.
           single_electron        - calcualte wavefronts from single electron.
           
Classes  : None.
"""

#-----------------------------------------------------------------------------#
# library

import random
import numpy as np
import scipy.sparse.linalg as ssl

from cat.utils import _multi
from cat.utils._srw_utils import _undulator
from cat.utils._srw_utils import _srw_electron_beam
from cat.utils._srw_utils import _propagate_wave_front

from cat.utils import _support
from cat.utils import _file_utils

#-----------------------------------------------------------------------------#
# constant

_N_SVD_TOP = 5000
_N_SVD_OPT = 2500
_N_SVD_TOL = 1000
_CUT_OFF = 1000

#-----------------------------------------------------------------------------#
# function
    
def _cal_wfrs(
    undulator, electron_beam, screen, file_name = None, method = 'srw'
    ):
    
    """
    ---------------------------------------------------------------------------
    calculate wavefronts from electron source to screens.
    
    args: undulator     - the parameters of undualtor.
          electron_beam - the parameters of electron_beam.
          screen        - the parameters of screen.
          file          - file name.
    
    return: none.
    ---------------------------------------------------------------------------
    """
    
    # The multi_process parameters
    
    c_rank = _multi._get_rank()
    n_rank = _multi._get_size()
    
    # The construction of undulator
    
    und = _undulator(undulator)
    und.wave_length()
    und.cal_k(electron_beam_energy = electron_beam["energy"])
    und_magnetic_structure = und.magnetic_structure()
    
    # The construction of electron beam
    
    e_beam = _srw_electron_beam(electron_beam, und.n_period, und.period_length)
    screen['screen'] = screen['screen'] - e_beam.initial_z/2
    e_beam.monte_carlo()
    
    # for 'vib'
    
    vir_part_beam, wavelength, resonance_energy = e_beam.after_monte_carlo(
        [0, 0, 0, 0, 0], und.period_length, und.k_vertical, und.k_horizontal
        )
    vir_wavefront = _propagate_wave_front(screen, resonance_energy)
    vir_wavefront._cal_wave_front(vir_part_beam, und_magnetic_structure) 
    vir_wavefront = np.reshape(vir_wavefront.wfr.arEx, (screen['nx'], screen['ny'], 2))
    vir_wavefront = vir_wavefront[:, :, 0] + 1j * vir_wavefront[:, :, 1]

    # The parameters of undulator and beam
    
    # reset number of electrons for the multi_layer_svd method
    n_electron = electron_beam['n_electron']

    if n_electron > _N_SVD_TOL * (n_rank - 1) and n_electron < _N_SVD_TOP * (n_rank - 1):
        pass
    else:
        n_electron = n_electron + (_N_SVD_OPT - n_electron % _N_SVD_OPT)

    n_points = screen['nx'] * screen['ny']
    electron_count, electron_index = _support._cal_rank_part(n_electron, n_rank)
    
    xtick = np.linspace(screen['xstart'], screen['xfin'], screen['nx'])
    ytick = np.linspace(screen['ystart'], screen['yfin'], screen['ny'])
    gridx, gridy = np.meshgrid(xtick, ytick)

    # Set monte carlo random seed
    
    random.seed(c_rank * 123)
    newSeed = random.randint(0, 1000000)
    random.seed(newSeed)

    #------------------------------------------------
    # wavefront calculation
    
    if c_rank > 0:
        
        n_electron_per_process = electron_count[c_rank - 1]
        e_crank_count, e_crank_index = _support._cal_part(
            n_electron_per_process, 500
            )
        
        _file_utils._create_multi_wfrs(
            c_rank, n_electron_per_process, n_points, file_name = file_name
            )
        
        flag = 0

        for i_part in e_crank_count:

            wfr_arrays = np.zeros((i_part, n_points), dtype = complex)

            for ie in range(i_part):

                rand_array = [random.gauss(0, 1) for ir in range(5)]
            
                e_beam.monte_carlo()
                part_beam, wavelength, resonance_energy = e_beam.after_monte_carlo(
                    rand_array, und.period_length, und.k_vertical, und.k_horizontal
                    )

                if method is 'srw':
            
                    # The magnetic structure setting  
                    
                    wavefront = _propagate_wave_front(screen, resonance_energy)
                    wavefront._cal_wave_front(part_beam, und_magnetic_structure)
                    wfr_arex = np.reshape(
                        np.array(wavefront.wfr.arEx), (screen['nx'], screen['ny'], 2)
                        )
                    wfr_arex = wfr_arex[:, :, 0] + 1j * wfr_arex[:, :, 1]

                    ie_wfr = np.reshape(wfr_arex, (1, n_points))
                
                elif method is 'vib':

                    offset_x = part_beam.partStatMom1.x
                    angle_x = part_beam.partStatMom1.xp
                    offset_y = part_beam.partStatMom1.y
                    angle_y = part_beam.partStatMom1.yp 

                    # phase shift

                    rot_x_phase = np.exp(
                        -1j * (2*np.pi/wavelength) * 
                        (angle_x * gridx - (1 - np.cos(angle_x)) * screen['screen'])
                        )
                    rot_y_phase = np.exp(
                        -1j * (2*np.pi/wavelength) * 
                        (angle_y * gridy - (1 - np.cos(angle_y)) * screen['screen'])
                        )

                    # offset range

                    offset_x = offset_x + np.sin(angle_x) * screen['screen']
                    offset_y = offset_y + np.sin(angle_y) * screen['screen']

                    # calculate range

                    x_00, x_01, x_10, x_11 = _support._shift_plane(
                        offset_x, xtick, 
                        screen['xstart'], screen['xfin'], 
                        screen['nx']
                        )
                    y_00, y_01, y_10, y_11 = _support._shift_plane(
                        offset_y, ytick, 
                        screen['ystart'], screen['yfin'], 
                        screen['ny']
                        )

                    shift_wfr = np.zeros((screen['nx'], screen['ny']), dtype = np.complex128)
                    shift_wfr[y_10 : y_11, x_10 : x_11] = (
                        (vir_wavefront * rot_x_phase * rot_y_phase)[y_00 : y_01, x_00 : x_01]
                        ) 
                    ie_wfr = np.reshape(shift_wfr, (1, n_points))

                wfr_arrays[ie, :] = ie_wfr
          
            # save wavefront
            
            _file_utils._save_multi_wfrs(
                    c_rank, wfr_arrays, 
                    e_crank_index[flag][0], e_crank_index[flag][-1] + 1, 
                    file_name
                    )
            flag += 1
            
        # source_file.close()
                
        _multi._send(np.array(1, dtype = 'i'), dtype = _multi.i, tag = c_rank)
        
        #------------------------------------------------ 
        
    if c_rank == 0:
        
        import os
        import h5py as h

        if os.path.isfile(file_name): os.remove(file_name)

        with h.File(file_name, 'a') as f: 

            descript = f.create_group("description")
            _support._dict_to_h5(descript, electron_beam)
            _support._dict_to_h5(descript, undulator)
            _support._dict_to_h5(descript, screen)
            descript.create_dataset("wavelength", data = wavelength)

        print("wavefront calculation start..... ", flush = True)
        
        for i in range(n_rank - 1):
            flag = _multi._recv(1, np_dtype = np.int, dtype = _multi.i, source = i + 1, tag = i + 1)
        
        print("wavefront calculation finished.", flush = True)




