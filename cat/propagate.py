#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_source_utils: Source construction support.

Functions: _fresnel    - DFFT fresenl propagation.
           propagate_s - single process coherent mode propagation
           propagate_m - multi-process coehrent mode propagation

Classes  : None
"""

#-----------------------------------------------------------------------------#
# library

import numpy as np

from cat.optics              import screen
from scipy                   import interpolate

# Warning! There is a bug in numpy.fft. Use scipy numpy fft instead.
# The pixel number of 2d arrray should be odd x odd. Or, half pxiel shift
# will be introudced. 
# For numpy, this shift was 1 and a half pxiel for odd x odd.
# Half pixel shift for even x even was oberved of both numpy fft and scipy fft.

from scipy                   import fft
from copy                    import deepcopy
from numpy                   import matlib 
#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function - propagators

def _fresnel_dfft(cmode_to_propagate,  wavelength, nx, ny,
                  xstart, ystart, xend, yend, rx, ry, distance):

# Warning! The fft2 of impulse function and ic should be done together with
# numpy fft fft2. Or some speckle will apear.

    """
    Double FFT Fresnel propagation of coherent mode.
    
    Args: cmode_to_propagate - the coherent mode to propagate.
          cmode_mask         - the mask of coherent mode.
          wavelength         - the wavelength of light field.
          nx                 - the dim of aixs x.
          ny                 - the dim of axis y.
          xstart             - location of start along axis x.
          ystart             - location of start along axis y.
          xend               - location of end along axis x.
          yend               - location of end along axis y.
          rx                 - the range of x.
          ry                 - the range of y.
          distance           - the distance of the propagation.
          
    Return: propagated coherent mode.
    """
    
    # wave number k
    wave_num = 2*np.pi / wavelength
    
    # the axis in frequency space
    qx = np.linspace(0.25/xstart, 0.25/xend, nx) * nx
    qy = np.linspace(0.25/ystart, 0.25/yend, ny) * ny
    
    mesh_qx, mesh_qy = np.meshgrid(qx, qy)
    
    # propagation function
    impulse_q = np.exp(
        (-1j * wave_num * distance) * 
        (1 - wavelength**2 * (mesh_qx**2 + mesh_qy**2))/2
        )
    
    # the multiply of coherent mode and propagation function
    propagated_cmode = fft.ifft2(
        fft.fft2(cmode_to_propagate) * 
        fft.ifftshift(impulse_q)
        )
      
    return propagated_cmode

def _bluestein_fft(g_input, fs, n, start, end, distance):
    
    n_ver, n_hor = np.shape(g_input)
    
    start_index = start + fs + 0.5 * (end - start) / n
    end_index = end + fs + 0.5 * (end - start) / n
    
    start_phase = np.exp(-1j * 2 * np.pi * start_index / fs)
    step_phase = np.exp(1j * 2 * np.pi * (end_index - start_index) / (n * fs))
    
    start_phase_neg_n = np.array(
        [start_phase**(-1*i) for i in range(n_ver)]
        )
    step_phase_n2 = np.array(
        [step_phase**(i**2/2) for i in range(n_ver)]
        )
    step_phase_k2 = np.array(
        [step_phase**(i**2/2) for i in range(n)]
        )
    step_phase_nk2 = np.array(
        [step_phase**(i**2/2) for i in range(-n_ver + 1, max(n_ver, n))]
        )
    step_phase_neg_nk2 = step_phase_nk2**(-1)
    
    fft_n = n_ver + n - 1
    count = 0
    while fft_n <= n_ver + n - 1:
        fft_n = 2**count
        count += 1
    
    conv_part0 = np.repeat(
        (start_phase_neg_n * step_phase_n2)[:, np.newaxis], 
        n_hor, axis = 1
        ) 
    conv_part1 = np.repeat(
        step_phase_neg_nk2[:, np.newaxis], 
        n_hor, axis = 1
        )
    
    conved = (
        fft.fft(g_input * conv_part0, fft_n, axis = 0) * 
        fft.fft(conv_part1, fft_n, axis = 0)
        )
    g_output = fft.ifft(conved, axis = 0)
    g_output = (
        g_output[n_ver : n_ver + n, :] * 
        np.repeat(step_phase_k2[:, np.newaxis], n_hor, axis = 1)
        )
    
    l = (end_index - start_index) * np.linspace(0, n - 1, n)/n + start_index
    shift_phase = matlib.repmat(
        np.exp(1j * 2 * np.pi * l * (-n_ver/2 + 0.5)/fs), n_hor, 1
        )
    g_output = g_output.T * shift_phase 

    return g_output 


def _asm_sfft(cmode_to_propagate, wavelength, nx, ny, xstart, ystart,
             xend, yend, rx, ry, distance):
    
    dx = (xstart - xend) / nx
    dy = (ystart - yend) / ny
    
    fx = np.linspace(-1/(2*dx), 1/(2*dx), nx)
    fy = np.linspace(-1/(2*dy), 1/(2*dy), ny)
    
    mesh_fx, mesh_fy = np.meshgrid(fx, fy)
    
    impulse = np.exp(
        -1j * 2 * np.pi * distance *
        np.sqrt(1 / wavelength**2 - (mesh_fx**2 + mesh_fy**2))
        )
    
    cmode_to_propagate = fft.ifftshift(fft.fft2(cmode_to_propagate))
    propagated_cmode = fft.ifft2(
        fft.ifftshift(impulse * cmode_to_propagate)
        )
    
    return propagated_cmode


def _fresnel_old_dfft(cmode_to_propagate,  wavelength, nx, ny,
                      xstart, ystart, xend, yend, rx, ry, distance):

# Warning! The fft2 of impulse function and ic should be done together with
# numpy fft fft2. Or some speckle will apear.

    """
    Double FFT Fresnel propagation of coherent mode.
    
    Args: cmode_to_propagate - the coherent mode to propagate.
          cmode_mask         - the mask of coherent mode.
          wavelength         - the wavelength of light field.
          nx                 - the dim of aixs x.
          ny                 - the dim of axis y.
          xstart             - location of start along axis x.
          ystart             - location of start along axis y.
          xend               - location of end along axis x.
          yend               - lcoation of end along axis y.
          rx                 - the range of x.
          ry                 - the range of y.
          distance           - the distance of the propagation.
          
    Return: propagated coherent mode.
    """
    
    # cmode_to_propagate = cmode_to_propagate
    
    # wave number k
    
    wave_num = 2*np.pi / wavelength
    
    # the axis in frequency space
    
    x0 = np.linspace(xstart, xend, nx)
    y0 = np.linspace(ystart, yend, ny)
    
    mesh_x, mesh_y = np.meshgrid(x0, y0)
    
    # propagation function
    
    impulse = (np.exp(1j * wave_num * distance) *
               np.exp(-1j * wave_num * (mesh_x**2 + mesh_y**2) / 
                      (2*distance)) /
               (1j * wavelength*distance))
    
    # the multiply of coherent mode and propagation function
    
    propagated_cmode = fft.ifftshift(
        fft.ifft2(fft.fft2(cmode_to_propagate) * 
        fft.fft2(impulse))
        )

    return propagated_cmode


def _kirchoff_fresnel(cmode_to_propagate, wavelength, 
                      fnx, fny, fxstart, fystart, fxend, fyend, fgridx, fgridy,
                      bnx, bny, bxstart, bystart, bxend, byend, bgridx, bgridy,
                      distance):

    count = bnx * bny
    xpixel = (fxend - fxstart)/fnx
    ypixel = (fyend - fystart)/fny
    
    front_wave = cmode_to_propagate
    back_wave = np.zeros((bny, bnx), dtype = complex).flatten()
    bgridx = bgridx.flatten()
    bgridy = bgridy.flatten()
    
    for i in range(count):
        
        print(i, flush = True)
        
        path = np.sqrt(
            (bgridx[i] - fgridx)**2 + (bgridy[i] - fgridy)**2 + distance**2
            )
        
        path_phase = 2*np.pi * path / wavelength
        costhe = distance / path
        
        back_wave[i] = back_wave[i] + np.sum(
            ((fxstart - fxend)/fnx) * ((fystart - fyend)/fny) *
            np.abs(front_wave) *
            
            # TO DO: The sign of phase in this package should be checked again!
            # 1*np.angle(front_wave) - path_phase
            
            np.exp(1j * (1 * np.angle(front_wave) - path_phase)) *
            costhe /
            (wavelength * path)
            )

    back_wave = np.reshape(back_wave, (bny, bnx))

    
    return back_wave 


# TO DO: Unstable !!!
def _bluestein_dft(wave, l0, l1, fs, count):
    
    #--------------------------------------
    # prepare parameters
    
    # The size of wavefront
    
    yc, xc = wave.shape
    
    print(wave.shape)
    
    # The range of outplane, calculated by the fourier propagation.
    
    r0 = l0 + (count*fs + (l1 - l0))/(2*count)
    r1 = l1 + (count*fs + (l1 - l0))/(2*count)
    
    # The parameters of chirp-z transform: z = a * w**-n
    # a: the start point in frequency range.
    # w: the range of frequency.
    
    a = np.exp(1j*2*np.pi*r0 / fs)
    w = np.exp(-1j*2*np.pi*(r1 - r0)/(count*fs))
    
    # The counting range of a and w
    
    k = np.arange(-yc, np.max([yc, count]))
    n = np.arange(0, yc)
    
    #--------------------------------------
    # calculation
    
    # calculate w**(k**2/2) and a**-n
    
    wk = w**(k**2/2)
    an = a**(-n)
    
    # calculate xn*(a**-n)*(w**(k**2/2))
    
    c0 = an * wk[yc + n -1]
    c0 = np.reshape(c0.repeat(xc), (yc, xc))
    c0 = np.multiply(wave, c0)
    
    #-------------------------------------
    # caluclate convolution by fft
    
    n2 = int(2**np.ceil(np.log2(yc + count - 2)))
    
    # calculate FT{w**(-k**2/2)}
    
    c1 = fft.fft(1/wk[0 : yc + count - 2], n2)
    c1 = np.reshape(c1.repeat(xc), (n2, xc))
    
    # calculate FT{xn*(a**-n)*(w**(k**2/2))}
    c2 = fft.fft(c0, n2, axis = 0)
    
    # calculate convlution of chirp-z transform
    
    c0 = fft.ifft(c1*c2, axis = 0)[yc : yc + count]
    wn = np.reshape(wk[yc : yc + count].repeat(xc), (count, xc))
    c0 = np.multiply(c0, wn)
    
    #------------------------------------
    # phase shift of fft
    
    l = np.arange(count) / count*(r1 - r0) + r0
    mshift = -yc/2
    
    import numpy.matlib as matlib
    
    mshift = np.matlib.repmat(
        np.exp(-1j*2*np.pi*l*(mshift + 0.5)/fs), xc, 1
        )
    
    wave = np.multiply(np.matrix(c0).T, np.matrix(mshift))
    
    return wave

# TO DO: Unstable !!!
def _bluestein(cmode_to_propagate, wavelength, 
              fnx, fny, fxstart, fystart, fxend, fyend, fgridx, fgridy,
              bnx, bny, bxstart, bystart, bxend, byend, bgridx, bgridy,
              distance):
   
    wave_num = 2 * np.pi / wavelength
    
    xpixel0 = (fxend - fxstart) / fnx
    ypixel0 = (fyend - fystart) / fny
    xpixel1 = (bxend - bxstart) / bnx
    ypixel1 = (byend - bystart) / bny
    
    fresnel0 = (
        np.exp(1j * wave_num * distance) * 
        np.exp(-0.5 * 1j * wave_num * (bgridx**2 + bgridy**2) / distance)
        )
    fresnel1 = (
        np.exp(-0.5 * 1j * wave_num * (fgridx**2 + fgridy**2) / distance)
        )
    g_input = cmode_to_propagate * fresnel1
    
    #------------------------------------------
    # propagate along y axis
    
    yfs = wavelength * distance / ypixel0
    yg_input = g_input
    
    g_input = _bluestein_fft(yg_input, yfs, bny, bystart, byend, distance)
    
    #------------------------------------------
    # propagate along x axis
    
    xfs = wavelength * distance / xpixel0
    xg_input = g_input
    
    g_input = _bluestein_fft(xg_input, xfs, bnx, bxstart, bxend, distance)
    
    #------------------------------------------
    # output
    
    g_output = g_input * fresnel0
    norm = (
        np.sum(np.abs(cmode_to_propagate)**2 * xpixel0 * ypixel0) / 
        np.sum(np.abs(g_output)**2 * xpixel1 * ypixel1)
        )
    
    return g_output * norm

#-----------------------------------------------------------------------------#
# function - propagation functions

def propagate_s(front, back, t = "fresnel"):
    
    """
    Single process: propgate coherent modes between front to back optics.
    
    Args: front - front optics
          back  - back optics

    Return: None.
    """
        
    # the distance of propagation. np.abs is not used, and back propagation is
    # supported.
    
    distance = back.position - front.position
     
    if distance == 0:
        
        for i in range(front.n):
            back.cmode[i] = back.cmode[i] * front.cmode[i]
            
    else:
        # the loop of every coherent mode
        
        for i in range(front.n):
            
            if t == "fresnel":
            
                back_cmode = _fresnel_dfft(
                    front.cmode[i], front.wavelength, 
                    front.xcount, front.ycount, front.xstart, front.ystart, 
                    front.xend, front.yend, front.xtick, front.ytick, 
                    distance
                    )
                
            elif t == "asm":
                
                back_cmode = _asm_sfft(
                    front.cmode[i], front.wavelength, 
                    front.xcount, front.ycount, front.xstart, front.ystart, 
                    front.xend, front.yend, front.xtick, front.ytick, 
                    distance
                    )
                     
            # normalize the propagated cmodes with the front optics
            # norm = (
            #     (np.sum(front.cmode[i])**2*front.xpixel*front.ypixel) /
            #     (np.sum(back_cmode)**2*back.xpixel*back.ypixel)
            #     )
            
            back.cmode[i] = back.cmode[i] * back_cmode

def propagate_old_s(front, back):
    
    """
    Single process: propgate coherent modes between front to back optics.
    
    Args: front - front optics
          back  - back optics

    Return: None.
    """
        
    # the distance of propagation. np.abs is not used, and back propagation is
    # supported.
    
    distance = back.position - front.position
     
    if distance == 0:
        
        for i in range(front.n):
            back.cmode[i] = back.cmode[i] * front.cmode[i]
            
    else:
        # the loop of every coherent mode
        
        for i in range(front.n):
            
            back_cmode = _fresnel_old_dfft(
                front.cmode[i], front.wavelength, 
                front.xcount, front.ycount, front.xstart, front.ystart, 
                front.xend, front.yend, front.xtick, front.ytick, 
                distance
                )
            
            # normalize the propagated cmodes with the front optics
            # norm = (
            #     (np.sum(front.cmode[i])**2*front.xpixel*front.ypixel) /
            #     (np.sum(back_cmode)**2*back.xpixel*back.ypixel)
            #     )
            
            back.cmode[i] = back.cmode[i] * back_cmode


def propagate_beamline(vectors, beamline):
    
    for i, vector in enumerate(vectors):
        
        if i == 0:
            recive = beamline(vector)
        else:
            i_recive = beamline(vector)
            recive.cmode.append(i_recive.cmode[0])

    return recive

# def propagte_n(front, back):
    
#     """
#     Near-field propagation. 
#     A trick is used here: 1. 
#     """

def propagate_k(front, back):
    
    distance = back.position - front.position
    
    for i in range(front.n):
        
        back_cmode = _kirchoff_fresnel(
            front.cmode[i], front.wavelength,
            front.xcount, front.ycount, front.xstart, front.ystart,
            front.xend, front.yend, front.gridx, front.gridy,
            back.xcount, back.ycount, back.xstart, back.ystart, back.xend,
            back.yend, back.gridx, back.gridy,
            distance
            )
        
        # normalize the propagated cmodes with the front optics
        # norm = (
        #     (np.sum(front.cmode[i])**2*front.xpixel*front.ypixel) /
        #     (np.sum(back_cmode)**2*back.xpixel*back.ypixel)
        #     )
        
        back.cmode[i] = back.cmode[i] * back_cmode

def propagate_czt(front, back):
    
    distance = back.position - front.position
    
    for i in range(front.n):
        
        back_cmode = _bluestein(
            front.cmode[i], front.wavelength,
            front.xcount, front.ycount, front.xstart, front.ystart,
            front.xend, front.yend, front.gridx, front.gridy,
            back.xcount, back.ycount, back.xstart, back.ystart, back.xend,
            back.yend, back.gridx, back.gridy,
            distance
            )
        
        back.cmode[i] = back.cmode[i] * back_cmode
        
def propagate_b(front, back):
    
    distance = back.position - front.position
    
    for i in range(front.n):
        
        back_cmode = _bluestein(
            front.cmode[i], front.wavelength,
            front.xcount, front.ycount, front.xstart, front.ystart,
            front.xend, front.yend, front.gridx, front.gridy,
            back.xcount, back.ycount, back.xstart, back.ystart, back.xend,
            back.yend, back.gridx, back.gridy,
            distance)
    
        back.cmode[i] = back.cmode[i] * back_cmode
                
#-----------------------------------------------------------------------------#