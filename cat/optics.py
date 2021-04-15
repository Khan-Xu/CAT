#-----------------------------------------------------------------------------#
# Copyright (c) 2021 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS HXS (B4) xuhan@ihep.ac.cn"
__date__     = "Date : 04.01.2021"
__version__  = "beta-1.0"


"""
optics: The optics.

Functions: None
           
Classes  : source     - the optics plane of coherent modes decomposition.
           screen     - a screen.
           crl        - crl. The default material is Be.
           KB         - KB mirror.
           ideal_lens - an ideal lens.
           rot_dcm    - the rocking effect of DCM.
           
"""

#-----------------------------------------------------------------------------#
# library

import numpy as np
import h5py  as h5

import pickle

from scipy             import interpolate
from copy              import deepcopy
from cat.utils._optics import _optic_plane, _optics
from cat.utils._optics import _locate

#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function

#-----------------------------------------------------------------------------#
# class

class source(_optics):
    
    """
    Construct the class of source.
    
    methods: 
    """
    
    def __init__(self, 
                 source = None, file_name = "test.h5", name = "source",
                 n_vector = 0, i_vector = None, position = 0):
        
        with h5.File(file_name, 'a') as f:
            
            # the geometry structure of the source plane
            
            self.xstart = np.array(f["description/xstart"])
            self.xend   = np.array(f["description/xfin"])
            self.xcount = int(np.array(f["description/nx"]))
            self.ystart = np.array(f["description/ystart"])
            self.yend   = np.array(f["description/yfin"])
            self.ycount = int(np.array(f["description/ny"]))
            
            self.location = np.array(f["description/screen"])
            num_vector = int(np.array(f["description/n_vector"]))
            
            # the cooridnate of wavefront
            
            self.xtick = np.linspace(self.xstart, self.xend, self.xcount)
            self.ytick = np.linspace(self.ystart, self.yend, self.ycount)
            
            self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
            
            self.xpixel = np.abs(self.xstart - self.xend)/self.xcount
            self.ypixel = np.abs(self.ystart - self.yend)/self.ycount
            
            self.mask = np.ones((self.ycount, self.xcount))
            self.plane = np.zeros((self.ycount, self.xcount), 
                                  dtype = complex)
            
            # the undulator parameters of source
            
            self.sigma_x0 = np.array(f["description/sigma_x0"])
            self.sigma_y0 = np.array(f["description/sigma_y0"])
            self.sigma_xd = np.array(f["description/sigma_xd"])
            self.sigma_yd = np.array(f["description/sigma_yd"])
            self.es       = np.array(f["description/energy_spread"])
            self.current  = np.array(f["description/current"])
            self.energy   = np.array(f["description/hormonic_energy"])
            self.n_electron = np.array(f["description/n_electron"])
            
            # the coherence properites of source.
            
            self.position   = self.location
            self.wavelength = np.array(f["description/wavelength"])
            self.ratio      = np.array(f["coherence/eig_value"])
            
            if i_vector is None:
                if n_vector:
                    self.n = n_vector
            else:
                self.n = 1
            
            self.name = name
            
            cmode = np.array(f["coherence/eig_vector"])
            
            if len(cmode.shape) == 2:
                cmode = np.reshape(cmode, 
                                   (self.ycount, self.xcount, num_vector))
                self.ori_cmode = [cmode[:, :, i] for i in range(num_vector)]
            elif len(cmode.shape) == 3:
                self.ori_cmode = [cmode[i, :, :] for i in range(num_vector)]
            
            if i_vector is None:
                self.cmode = [self.ori_cmode[i] for i in range(self.n)]
            else:
                self.cmode = [self.ori_cmode[i_vector]]
                
            #---------------------------------------------------
            # the 1d and 2d moi calculation
            
            self.csd2dy = np.array(f["coherence/csdx"])
            self.csd2dx = np.array(f["coherence/csdy"])
            self.csd1dx = np.abs(np.diag(np.fliplr(self.csd2dx)))
            self.csd1dy = np.abs(np.diag(np.fliplr(self.csd2dy)))
            
            # cal spectral degree of coherence
            
            i0x = np.diag(np.abs(self.csd2dx))
            i0y = np.diag(np.abs(self.csd2dy))
            
            i0x2d = np.zeros((self.xcount, self.xcount))
            i0y2d = np.zeros((self.ycount, self.ycount))
            
            for i in range(self.xcount): i0x2d[i, :] = i0x
            for i in range(self.ycount): i0y2d[i, :] = i0y
            
            self.miu2dx = self.csd2dx / np.sqrt(i0x2d * i0x2d.T)
            self.miu2dy = self.csd2dy / np.sqrt(i0y2d * i0y2d.T)
            
            self.miu1dx = np.abs(np.diag(np.fliplr(self.miu2dx)))
            self.miu1dy = np.abs(np.diag(np.fliplr(self.miu2dy)))
            
            # the coherent length along x and y axes
            
            self.clx = float()
            self.cly = float()

#------------------------------------------------------------------------------

class screen(_optics):
    """
    Construct the class of screen.
    
    methods: 
    """
    
    def __init__(self,
                 optics = None, n = 0, location = 0):
        
        super().__init__(optic = optics, name = "screen", n_vector = n, 
                         position = location)
        
        self.cmode = [self.mask for i in range(self.n)]

#------------------------------------------------------------------------------
           
class crl(_optics):
    """
    Construct the class of crl.
    
    methods: 
    """
    
    def __init__(self, 
                 optics = None, n = 0, location = 0,
                 nlens = 0, delta = 3.41e-6, rx = 0, ry = 0):
    
        super().__init__(optic = optics, name = "crl", 
                         n_vector = n, 
                         position = location)
        
        self.focus_x = rx/(2*nlens*delta) if rx != 0 else 1e20
        self.focus_y = ry/(2*nlens*delta) if ry != 0 else 1e20
        
        #---------------------------------------------------
        # add vibration of source
        
        self.lens_phase = np.exp(
            1j*(2*np.pi/self.wavelength) *
            (self.gridx**2/(2*self.focus_x) + 
             self.gridy**2/(2*self.focus_y))
            )
         
        #---------------------------------------------------
        
        for i in range(self.n):
            self.cmode.append(self.lens_phase)

#------------------------------------------------------------------------------

class KB(_optics):
    """
    Construct the class of KB mirror. The effect of rocking and offset were
    considered.
    
    methods: 
    """
    
    def __init__(self, 
                 optics = None, direction = 'v', 
                 n = 0, location = 0,
                 pfocus = 0, qfocus = 0, length = 1, angle = 0,
                 e = None):
    
        super().__init__(optic = optics, name = "KB", 
                         n_vector = n, 
                         position = location)
        
        if direction == 'h':
            self.lens_phase = np.exp(
                1j*(2*np.pi/self.wavelength) *
                (np.sqrt(self.gridx**2 + pfocus**2) +
                 np.sqrt(self.gridx**2 + qfocus**2))
                )
        
        elif direction == 'v':
            self.lens_phase = np.exp(
                1j*(2*np.pi/self.wavelength) *
                (np.sqrt(self.gridy**2 + pfocus**2) +
                 np.sqrt(self.gridy**2 + qfocus**2))
                )
    
        for i in range(self.n):
            self.cmode[i] = self.lens_phase

#------------------------------------------------------------------------------
    
class ideal_lens(_optics):
    """
    Construct the class of ideal lens
    
    methods
    """
    
    def __init__(self, optics = None, n = 0, location = 0,
                 xfocus = 0, yfocus = 0, e = None):
        
        super().__init__(optic = optics, name = "ideal lens", 
                         n_vector = n, 
                         position = location)
        
        self.focus_x = xfocus
        self.focus_y = yfocus
        
        self.lens_phase = np.exp(
            1j*(2*np.pi/self.wavelength) *
            (self.gridx**2/(2*self.focus_x) + 
             self.gridy**2/(2*self.focus_y))
            )
                
        if e is None:
            error_phase = 1
        else:
            error_phase = np.exp(1j*e)
            
        for i in range(self.n):
            self.cmode[i] = self.lens_phase
            
#------------------------------------------------------------------------------
