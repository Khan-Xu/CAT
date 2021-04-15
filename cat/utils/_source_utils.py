#-----------------------------------------------------------------------------#
# Copyright (c) 2021 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS HXS (B4) xuhan@ihep.ac.cn"
__date__     = "Date : 04.01.2021"
__version__  = "beta-1.0"


"""
_source_utils: Source construction support.

Functions: None.
           
Classes  : _srw_electron_beam    - construct e_beam base on monte carlo method.
           _undulator            - undulator parameters setting.
           _propagete_wave_front - propagate wavafront from source to screen.
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
    Divide n_tot into seveal parts with the length n_divide.
    For example: n_tot - 11, n_divide - 3 result - count [3, 3, 3, 2], 
                 index [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
    
    Args: n_tot    - total number to divide.
          n_divide - the length of the part.

    Return: count - a list contains the length of every part.
            index - a list contains the index of every part.
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
    Divide n_tot into n_rank - 1 parts. 
    Used to plan the distribution of multiprocess. Usullay, rank == 0 is not 
    used for the distribution plan.
    For example: n_tot - 11, n_rank - 4, result - count [3, 3, 3, 2], 
                 index [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
    
    Args: n_tot  - total number to divide.
          n_rank - the total rank number.

    Return: count - a list contains the length of every part.  
            index - a list contains the index of every part.
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
    Save python dict to a h5py group with dataset.
    
    Args: group     - h5py group.
          dict_file - python dict to save.
         
    Return: None.
    """
    
    for key, value in dict_file.items():
        group.create_dataset(key, data = value)
        
def _require_h5file(file_name):
    
    """
    Create a h5py file, remove it and create a new one if exist.
    
    Args: file_name - the name of the h5py file.
         
    Return: h5file handle.
    """
    
    if os.path.isfile(file_name): os.remove(file_name)
    
    return h.File(file_name, 'a')

# Todo: delete it or develop it.
    
def _h5data_combine(dataset, array_data, dim = 0):
    
    shape_new = array_data.shape[dim]
    shape_old = dataset.shape[dim]
    
    dataset.resize((shape_new + shape_old, ))
    dataset[shape_old : shape_old + shape_new, ] = array_data


def _create_multi_source_file(i_rank, n_electron, n_points, 
                              file_name = "test.h5"):
    
    """
    Create a cache h5py file of source.
    
    Args: i_rank     - the root source file is constructed in rank == 0, 
                       the cache file is created in rank > 0 processes.
          n_electron - the number of electron to put.
          n_points   - the pixel number of screen.
          file_name  - 
         
    Return: None.
    """
    
    # create cache file in rank > 0
    
    if i_rank > 0:
        file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (i_rank)
    file = _require_h5file(file_name)
    
    # create group to store number of electrons in this process
    
    index = file.create_group("index")
    index.create_dataset("n_electron_per_process", data = n_electron)
    
    # create group to store the phase space of electrons
    
    phase = file.create_group("phase_space")
    phase_array = np.zeros(n_electron, dtype = 'float')
    for item in ['x0', 'xp', 'y0', 'yp', 'e0']:
        phase.create_dataset(item, data = np.copy(phase_array))
    
    # create group to store mesh of the screen
        
    screen_mesh = file.create_group("screen_mesh")
    mesh_array = np.zeros(n_electron, dtype = 'float')
    for item in ['xl', 'xr', 'nx', 'yl', 'yr', 'ny']:
        screen_mesh.create_dataset(item, data = np.copy(mesh_array))
    
    # create group to store the wave_fronts
        
    wave_front = file.create_group("wave_front")
    wfr_array = np.zeros((n_electron, n_points), dtype = 'complex')
    wave_front.create_dataset("arex", data = np.copy(wfr_array))
    wave_front.create_dataset("arey", data = np.copy(wfr_array))
    
    file.close()

def _save_source_file(i, i_rank, n_points, file, part_beam, wavefront, 
                      screen_nx, screen_ny, resonance_energy,):
    
    """
    Save calcualte wavefront to a cache file.
    
    Args: i              - the loop index.
          i_rank         - the root source file is constructed in rank == 0, 
                           the cache file is created in rank > 0 processes.
          resonce_energy - ligth energy to save.
          n_points       - the pixel number of screen.
          file           - file name of root source file.
          part_beam      - the electron particle beam.
          wave_front     - wavefront to save.
         
    Return: None.
    """
    
    source_file = h.File(
        ('_' + file.split('.')[0] + '_%.2d.h5') % (i_rank), 'a'
        )
    
    # save phase space
    
    phase = source_file["phase_space"]
    phase["x0"][i] = part_beam.partStatMom1.x
    phase["xp"][i] = part_beam.partStatMom1.xp
    phase["y0"][i] = part_beam.partStatMom1.y
    phase["yp"][i] = part_beam.partStatMom1.yp
    phase["e0"][i] = resonance_energy
    
    # save screen mesh
    
    screen = source_file["screen_mesh"]
    screen["xl"][i] = wavefront.mesh.xStart
    screen["xr"][i] = wavefront.mesh.xFin
    screen["nx"][i] = wavefront.mesh.nx
    screen["yl"][i] = wavefront.mesh.yStart
    screen["yr"][i] = wavefront.mesh.yFin
    screen["ny"][i] = wavefront.mesh.ny
    
    # save wavefront
    
    wave_front = source_file["wave_front"]
    arex = np.array(wavefront.wfr.arEx)
    arey = np.array(wavefront.wfr.arEy)
    
    # the wavefront array calucated by srw is a litte wired.
    
    arex = np.reshape(arex, (screen_nx, screen_ny, 2))
    arey = np.reshape(arey, (screen_nx, screen_ny, 2))
    arex = arex[:, :, 0] + 1j * arex[:, :, 1]
    arey = arey[:, :, 0] + 1j * arey[:, :, 1]
    wave_front["arex"][i, :] = np.reshape(arex, (1, screen_nx * screen_ny))
    wave_front["arey"][i, :] = np.reshape(arey, (1, screen_nx * screen_ny))
    
    source_file.close()
    
def _reconstruct_source_file(file_name, n_electron, n_points, electron_beam, 
                             undulator, screen, wavelength, n_rank):
    
    """
    Cosntruct root source file from the cachle source file from every process.
    
    Args: file_name     - 
          n_electron    - the number of electron to put.
          n_points      - the pixel number of screen.
          electron_beam - parameters of e_beam.
          undulator     - parameters of undualtor.
          screen        - parameters of screen.
          n_rank        - total rank number.

    Return: None.
    """
    
    source_file = _require_h5file(file_name)
    
    # save parameters of e_beam, undualtor and screen.
    
    descript = source_file.create_group("description")
    _dict_to_h5(descript, electron_beam)
    _dict_to_h5(descript, undulator)
    _dict_to_h5(descript, screen)
    descript.create_dataset("wavelength", data = wavelength)
    
    # create phase space group
    
    phase_array = np.zeros(n_electron, dtype = 'float')
    phase_space_x0 = np.copy(phase_array)
    phase_space_xp = np.copy(phase_array)
    phase_space_y0 = np.copy(phase_array)
    phase_space_yp = np.copy(phase_array)
    phase_space_e0 = np.copy(phase_array)
    
    # create screen mesh group
    
    mesh_array = np.zeros(n_electron, dtype = 'float')
    screen_mesh_xl = np.copy(mesh_array)
    screen_mesh_xr = np.copy(mesh_array)
    screen_mesh_nx = np.copy(mesh_array)
    screen_mesh_yl = np.copy(mesh_array)
    screen_mesh_yr = np.copy(mesh_array)
    screen_mesh_ny = np.copy(mesh_array)
    
    # create wavefront group
    
    wfr_array = np.zeros((n_electron, n_points), dtype = 'complex')
    wfr_arex = np.copy(wfr_array)
    wfr_arey = np.copy(wfr_array)
    
    count = 0
    
    # loading data from every cache data
    
    for i in range(n_rank - 1):
        
        i_file_name = ('_' + file_name.split('.')[0] + '_%.2d.h5') % (i + 1)
        i_file = h.File(i_file_name, 'a')
        
        ine = np.array(i_file["index/n_electron_per_process"])
        end = count + ine
        
        phase_space_x0[count : end] = np.array(i_file["phase_space/x0"])
        phase_space_xp[count : end] = np.array(i_file["phase_space/xp"])
        phase_space_y0[count : end] = np.array(i_file["phase_space/y0"])
        phase_space_yp[count : end] = np.array(i_file["phase_space/yp"])
        phase_space_e0[count : end] = np.array(i_file["phase_space/e0"])
        
        screen_mesh_xl[count : end] = np.array(i_file["screen_mesh/xl"])
        screen_mesh_xr[count : end] = np.array(i_file["screen_mesh/xr"])
        screen_mesh_nx[count : end] = np.array(i_file["screen_mesh/nx"])
        screen_mesh_yl[count : end] = np.array(i_file["screen_mesh/yl"])
        screen_mesh_yr[count : end] = np.array(i_file["screen_mesh/yr"])
        screen_mesh_ny[count : end] = np.array(i_file["screen_mesh/ny"])
        
        wfr_arex[count : end, :] = np.array(i_file["wave_front/arex"])
        wfr_arey[count : end, :] = np.array(i_file["wave_front/arey"])
        
        count = count + ine
        if os.path.isfile(i_file_name): os.remove(i_file_name)
    
    phase = source_file.create_group("phase_space")
    phase.create_dataset("x0", data = phase_space_x0)
    phase.create_dataset("xp", data = phase_space_xp)
    phase.create_dataset("y0", data = phase_space_y0)
    phase.create_dataset("yp", data = phase_space_yp)
    phase.create_dataset("e0", data = phase_space_e0)
    
    screen_mesh = source_file.create_group("screen_mesh")
    screen_mesh.create_dataset("xl", data = screen_mesh_xl)
    screen_mesh.create_dataset("xr", data = screen_mesh_xr)
    screen_mesh.create_dataset("nx", data = screen_mesh_nx)
    screen_mesh.create_dataset("yl", data = screen_mesh_yl)
    screen_mesh.create_dataset("yr", data = screen_mesh_yr)
    screen_mesh.create_dataset("ny", data = screen_mesh_ny)
    
    with h.File("_cache.h5", 'a') as f:
        
        wave_front = f.create_group("wave_front")
        wave_front.create_dataset("arex", data = wfr_arex)
        wave_front.create_dataset("arey", data = wfr_arey)
    
    
def _cal_csd(n_electron, n_points, nx, ny, file_name = "test.h5"):
    
    """
    Calculate csd from wavefronts.
    
    Args: n_electron - the number of electron to put.
          n_points   - the pixel number of screen.
          file_name  - 

    Return: None.
    """

    print("| csd calculation start.          |", flush = True)
    
    with h.File("_cache.h5", 'a') as f:
        
        wfr = np.array(f["wave_front/arex"])
        
        # csd is calucated with seveal parts.
        
        J_count, J_index = _cal_rank_part(n_points, _N_Part)
        
        # create group to save csd
        
        coherence = f.create_group("coherence")
        coherence.create_dataset("csd", (0, 0), dtype = np.complex128, 
                                  maxshape = (None, None))
        
        # slice csd along axis x and y
        
        if nx % 2:
            cx = int(nx//2)
            wfry = np.reshape(wfr, [n_electron, nx, ny])[:, cx, :]
        else:
            cx = [int(nx//2 - 1), int(nx//2)]
            wfry = (
                np.reshape(wfr, [n_electron, nx, ny])[:, cx[0], :] +
                np.reshape(wfr, [n_electron, nx, ny])[:, cx[1], :]
                )/2

        if ny % 2:
            cy = int(ny//2)
            wfrx = np.reshape(wfr, [n_electron, nx, ny])[:, :, cy]
        else:
            cy = [int(ny//2 - 1), int(ny//2)]
            wfrx = (
                np.reshape(wfr, [n_electron, nx, ny])[:, :, cy[0]] +
                np.reshape(wfr, [n_electron, nx, ny])[:, :, cy[1]]
                )/2

        # csdx and csdy is calcualted.
        
        csdx = np.dot(wfrx.T.conj(), wfrx)
        csdy = np.dot(wfry.T.conj(), wfry)
        coherence.create_dataset("csdx", data = csdx)
        coherence.create_dataset("csdy", data = csdy)
        
        # miux and miuy is calcualted base on the intensity
        
        miux = csdx / np.abs(np.diagonal(csdx))
        miuy = csdy / np.abs(np.diagonal(csdy))
        coherence.create_dataset("miux", data = miux)
        coherence.create_dataset("miuy", data = miuy)
        
        # calcuate and store the whole csd
        
        for ix, i_Jx in enumerate(J_index):
            
            # resize csd to save more.
            
            x_old, x_new = sum(J_count[0 : ix]), sum(J_count[0 : ix + 1])
            coherence["csd"].resize((x_new, sum(J_count)))
            
            # ToDo: if new cmd method is used
            
            # J = _require_h5file("_J%.2d.h5" % (ix + 1))
            # J.create_group("J")
            # J["J"].create_dataset("J_part", (J_count[ix], sum(J_count)),
            #                   dtype = np.complex128, 
            #                   maxshape = (None, None))
            
            for iy, i_Jy in enumerate(J_index):
                
                y_old, y_new = sum(J_count[0 : iy]), sum(J_count[0 : iy + 1])
                
                wfr_x = wfr[:, i_Jx]
                wfr_y = wfr[:, i_Jy]
                
                csd_part = np.dot(wfr_x.T.conj(), wfr_y)
                coherence["csd"][x_old : x_new, y_old : y_new] = csd_part  
                
                # J["J/J_part"][:, y_old : y_new] = csd_part 
                
            # J.close()
            
            # print every loop
            
            print("| _J%.2d saved                      |" % 
                  (ix + 1), flush = True)

    print("| csd calculation finished.       |", flush = True)
    print("___________________________________", flush = True)
    
def _scipy_sparse_cmd(n_vector = 500, file_name = "test.h5"):
    
    """
    CSD CMD using scipy.sparse.linalg.eigsh.
    
    Args: n_vector  - the number of vectors to approxmite.
          file_name - 

    Return: None.
    """
        
    print("| coherent mode calculation start.|", flush = True)
    
    with h.File("_cache.h5", 'a') as f:
        
        # load csd, time costing.
        
        csd = np.array(f["coherence/csd"])
        csdx = np.array(f["coherence/csdx"])
        csdy = np.array(f["coherence/csdy"])
        miux = np.array(f["coherence/miux"])
        miuy = np.array(f["coherence/miuy"])
        
        # load linalg.eigsh.
        
        from scipy.sparse import linalg
        eig_value, eig_vector = linalg.eigsh(csd, k = n_vector)
        
        # saving reuslts.
    
    if os.path.isfile("_cache.h5"): os.remove("_cache.h5")
    
    with h.File(file_name, 'a') as f:
        
        coherence = f.create_group("coherence")
        f["coherence"].create_dataset("eig_vector", data = eig_vector)
        f["coherence"].create_dataset("eig_value", data = eig_value)
        f["coherence"].create_dataset("csdx", data = csdx)
        f["coherence"].create_dataset("csdy", data = csdy)
        f["coherence"].create_dataset("miux", data = miux)
        f["coherence"].create_dataset("miuy", data = miuy)
        
        
    print("| coherent mode calculated.       |", flush = True)
    print("___________________________________", flush = True)
    
def _arnoldi_cmd(n_points, n_vector = 500, file_name = "test.h5"):
    
    """
    CSD CMD arnoldi method.
    
    Args: n_points  - the size (one dimension) of matrix
          n_vector  - the number of vectors to approxmite.
          file_name - 

    Return: None.
    """
    
    # The multi process parameters
    
    n_rank = _multi._get_size()
    rank = _multi._get_rank()
    
    # multi-process distribution plan
    
    count, index = _cal_rank_part(n_points, n_rank)
    order = 2*n_vector + 1
    
    # construct guess_vector, shcur and hess matrix. loading csd matrix and 
    # distrub csd parts to different multi-process
    
    if rank == 0:
        
        print("| coherent mode calculation start.|", flush = True)
        
        # construct guess_vector and normalise.
        
        guess_vector = np.random.random(n_points)
        guess_vector = guess_vector / np.linalg.norm(guess_vector)
        
        schur = np.zeros((order, n_points), dtype = np.complex128)
        schur_conj = np.copy(schur)
        schur[0, :] = guess_vector
        schur_conj[0, :] = schur[0, :].conj()
        
        hess = np.zeros((order, order), dtype = np.complex128)
        
        with h.File("_cache.h5", 'a') as f:
            
            csdx = np.array(f["coherence/csdx"])
            csdy = np.array(f["coherence/csdy"])
            miux = np.array(f["coherence/miux"])
            miuy = np.array(f["coherence/miuy"])
            
            print("| csd data loading...             |", flush = True)
            
            for ic in range(n_rank - 1):
                csd_part = np.array(f["coherence/csd"][index[ic], :])
                _multi._send(csd_part, 
                             dtype = _multi.c, 
                             dest = ic + 1, 
                             tag = 0)
                del csd_part
    
    # Recving csd parts
                
    elif rank > 0:
        
        csd_part = _multi._recv(
            (count[rank - 1], n_points), np_dtype = np.complex128, 
            dtype = _multi.c, source = 0, tag = 0
            )
        
        if rank == n_rank - 1:
            print("| all csd data loaded.            |", flush = True)
            print("___________________________________", flush = True)
        else:
            print("| rank%.2d csd loaded.              |" % (rank), 
                  flush = True)
    
    # start arnoldi loop
            
    for i in range(1, order):
        
        # for rank >0 process, A*q, A2*q, A3q .... is calculated.
        
        if rank > 0:
            
            vector = _multi._recv(n_points, 
                                  np_dtype = np.complex128, 
                                  dtype = _multi.f, 
                                  source = 0, 
                                  tag = 10*i + 1)
            vector_to_send = np.dot(csd_part, vector)
            _multi._send(vector_to_send, dtype = _multi.c, dest = 0, tag = i)
        
        # GSM process
            
        if rank == 0:
            
            if (10*i) % (10*int(order/10)) == 0:
                print("| %.2d percent                      |" % 
                      (int(100*i/order)), flush = True)
            
            for ir in range(n_rank - 1):
                _multi._send(
                    schur[i-1, :], dtype = _multi.c, 
                    dest = ir + 1, tag = 10*i + 1
                    )
            
            idx = 0
            
            for ir in range(n_rank - 1):
                vector_part = _multi._recv(
                    count[ir], np_dtype = np.complex128, dtype = _multi.c, 
                    source = ir + 1, tag = i
                    )
                schur[i, int(idx) : int(idx + count[ir])] = vector_part
                idx += count[ir]
                
            for ih in range(i):
                hess[ih, i - 1] = schur_conj[ih, :].dot(schur[i, :])
                schur[i, :] = schur[i, :] - hess[ih, i - 1]*schur[ih, :]
            
            hess[i, i - 1] = np.linalg.norm(schur[i, :])
            schur[i, :] = schur[i, :] / hess[i, i - 1]
            schur_conj[i, :] = schur[i, :].conj()
    
    # calculate approximate value and vector
            
    if rank == 0:
        
        hess = hess[0 : order, 0 : order]
        schur = schur[0 : order, :]
        ratio = np.linalg.eigh(hess)
        
        eig_values  = ratio[0][: : -1]
        vector = ratio[1].transpose()[: : -1, :]
        vector = vector.transpose()
        
        schur_vector = np.zeros((n_points, n_vector), dtype = np.complex128)
        
        for it in range(n_vector):
            t = vector[:, it]
            schur_vector[:, it] = schur.transpose().dot(t)
            
        with h.File(file_name, 'a') as f:
            
            if os.path.isfile("_cache.h5"): os.remove("_cache.h5")
            
            coherence = f.create_group("coherence")
            f["coherence"].create_dataset("eig_vector", data = schur_vector)
            f["coherence"].create_dataset("eig_value", data = eig_values)
            f["coherence"].create_dataset("csdx", data = csdx)
            f["coherence"].create_dataset("csdy", data = csdy)
            f["coherence"].create_dataset("miux", data = miux)
            f["coherence"].create_dataset("miuy", data = miuy)
        
        print("| coherent mode calculated.       |", flush = True)
        print("___________________________________", flush = True)
                 
#-----------------------------------------------------------------------------#
