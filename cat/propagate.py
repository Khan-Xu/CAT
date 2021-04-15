#-----------------------------------------------------------------------------#
# Copyright (c) 2021 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS HXS (B4) xuhan@ihep.ac.cn"
__date__     = "Date : 04.01.2021"
__version__  = "beta-1.0"


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

from cat.utils._source_utils import _cal_rank_part
from cat.optics              import screen
from cat.utils               import _multi
from scipy                   import interpolate

# Warning! There is a bug in numpy.fft. Use scipy numpy fft instead.
# The pixel number of 2d arrray should be odd x odd. Or, half pxiel shift
# will be introudced. 
# For numpy, this shift was 1 and a half pxiel for odd x odd.
# Half pixel shift for even x even was oberved of both numpy fft and scipy fft.

from scipy                   import fft
from copy                    import deepcopy
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
    
    """
    kirchoff fresnel diffraction of coherent mode.
    
    Args: cmode_to_propagate - the coherent mode to propagate.
          wavelength         - the wavelength of light field.
          fnx                - the dim of aixs x of front plane.
          fny                - the dim of axis y of front plane.
          fxstart            - location of start along axis x of front plane.
          fystart            - location of start along axis y of front plane.
          fxend              - location of end along axis x of front plane.
          fyend              - lcoation of end along axis y of front plane.
          fgridx             - the meshgrid of x of front plane.
          fgridy             - the meshgrid of y of front plane.
          bnx                - the dim of aixs x of back plane.
          bny                - the dim of axis y of back plane.
          bxstart            - location of start along axis x of back plane.
          bystart            - location of start along axis y of back plane.
          bxend              - location of end along axis x of back plane.
          byend              - lcoation of end along axis y of back plane.
          bgridx             - the meshgrid of x of back plane.
          bgridy             - the meshgrid of y of back plane.
          distance           - the distance of the propagation.
          
    Return: propagated coherent mode.
    """
    
    count = bnx * bny
    xpixel = (fxend - fxstart)/fnx
    ypixel = (fyend - fystart)/fny
    
    front_wave = cmode_to_propagate
    back_wave = np.zeros((bny, bnx), dtype = complex).flatten()
    bgridx = bgridx.flatten()
    bgridy = bgridy.flatten()
    
    for i in range(count):
        
        path = np.sqrt(
            (bgridx[i] - fgridx)**2 + (bgridy[i] - fgridy)**2 + distance**2
            )
        
        path_phase = 2*np.pi * path / wavelength
        costhe = distance / path
        
        back_wave[i] = back_wave[i] + np.sum(
            ((fxstart - fxend)/fnx) * 
            ((fystart - fyend)/fny) *
            np.abs(front_wave) *
            
            # Todo: The sign of phase in this package should be checked again!
            # -1*np.angle(front_wave)
            
            np.exp(1j * (-1 * np.angle(front_wave) + path_phase)) *
            costhe /
            (wavelength * path)
            )
    back_wave = np.reshape(back_wave, (bny, bnx))
    
    return back_wave

#-----------------------------------------------------------------------------#
# function - propagation functions

def propagate_s(front, back):
    
    """
    Single process: propgate coherent modes between front to back optics.
                    Using Fresnel diffraction.
    
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
            
            back_cmode = _fresnel_dfft(
                front.cmode[i], front.wavelength, 
                front.xcount, front.ycount, front.xstart, front.ystart, 
                front.xend, front.yend, front.xtick, front.ytick, 
                distance
                )
        
            back.cmode[i] = back.cmode[i] * back_cmode

def propagate_beamline(vectors, beamline):
    
    """
    Single process: propgate coherent modes between front to back optic. 
                    Support the propagation of specified coherent mode.
    
    Args: vector   - the sequence number of coherent mode to propagate.
          beamline - the function of beamline. 

    Return: the returned beamline function of beamline.
    """
    
    for i, vector in enumerate(vectors):
        
        if i == 0:
            recive = beamline(vector)
        else:
            i_recive = beamline(vector)
            recive.cmode.append(i_recive.cmode[0])

    return recive

def propagate_k(front, back):
    
    """
    Single process: propgate coherent modes between front to back optic. 
                    Using kirchhoff fresnel diffraction.
    
    Args: front - front optics
          back  - back optics

    Return: None.
    """
    
    distance = back.position - front.position
    
    for i in range(front.n):
        
        back_cmode = _kirchoff_fresnel(
            front.cmode[i], front.wavelength,
            front.xcount, front.ycount, front.xstart, front.ystart,
            front.xend, front.yend, front.gridx, front.gridy,
            back.xcount, back.ycount, back.xstart, back.ystart, back.xend,
            back.yend, back.gridx, back.gridy,
            distance)
    
        back.cmode[i] = back.cmode[i] * back_cmode
        
#-----------------------------------------------------------------------------#