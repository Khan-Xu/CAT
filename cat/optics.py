#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_source_utils: Source construction support.

Functions: None
           
Classes  : _optic_plane - the geometry structure of optics
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

# functions of x-ray responase of rocking
#-----------------------------------------------------------------------------#
# class

class source(_optics):
    
    """
    Construct the class of source.
    
    methods: 
    """
    
    def __init__(self, 
                 source = None, file_name = "test.h5", name = "source",
                 n_vector = 0, i_vector = None, position = 0,
                 offx = 0, offy = 0, rotx = 0, roty = 0):
        
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
            # add rotation of source
            
            rotx_phase = np.exp(
                -1j*(2*np.pi/self.wavelength)*
                (rotx*self.gridx - (1 - np.cos(rotx))*self.position)
                )
            roty_phase = np.exp(
                -1j*(2*np.pi/self.wavelength)*
                (roty*self.gridy - (1 - np.cos(roty))*self.position)
                )                
            #---------------------------------------------------
            # add vibration of the source
    
            offx = offx + np.sin(rotx) * self.position
            offy = offy + np.sin(roty) * self.position
            
            if offx > 0:
                loclx0 = _locate(self.xtick, self.xstart + offx)
                locrx0 = self.xcount - loclx0
                loclx1 = 0
                locrx1 = self.xcount - 2*loclx0
                
            elif offx <= 0:
                locrx0 = _locate(self.xtick, self.xend + offx)
                loclx0 = self.xcount - locrx0
                loclx1 = self.xcount - 2*locrx0
                locrx1 = self.xcount
            
            if offy > 0:
                locly0 = _locate(self.ytick, self.ystart + offy)
                locry0 = self.ycount - locly0
                locly1 = locly0
                locry1 = self.ycount - locly0
                
            elif offy <= 0:
                locry0 = _locate(self.ytick, self.yend + offy)
                locly0 = self.ycount - locry0
                locly1 = self.ycount - locry0
                locry1 = locry0
                
            for i in range(self.n):
            
                plane = deepcopy(self.plane)
                plane[:, loclx0 : locrx0] = (
                    (self.cmode[i] * 
                     rotx_phase * roty_phase)[:, loclx1 : locrx1]
                    )
                self.cmode[i] = plane
                
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
            
class source2(_optics):
    
    """
    Construct the class of source.
    
    methods: 
    """
    
    def __init__(self, 
                 source = None, file_name = "test.h5", name = "source",
                 n_vector = 0, i_vector = None, position = 0,
                 offx = 0, offy = 0, rotx = 0, roty = 0):
        
        with h5.File(file_name, 'a') as f:
            
            # the geometry structure of the source plane
            
            self.xstart = np.array(f["description/xstart"])
            self.xend   = np.array(f["description/xfin"])
            self.xcount = int(np.array(f["description/nx"]))
            self.ystart = np.array(f["description/ystart"])
            self.yend   = np.array(f["description/yfin"])
            self.ycount = int(np.array(f["description/ny"]))
            
            self.location = np.array(f["description/screen"])
            num_vector = 2 * int(np.array(f["description/n_vector"]))
            
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
            self.ratio      = np.array(f["coherence/eig_value"])**2
            
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
                self.cmode = [
                    np.abs(self.ori_cmode[i]) * np.exp(-1j * np.angle(self.ori_cmode[i])) 
                    for i in range(self.n)
                    ]
            else:
                self.cmode = [self.ori_cmode[i_vector]]
            
            #---------------------------------------------------
            # add rotation of source
            
            rotx_phase = np.exp(
                -1j*(2*np.pi/self.wavelength)*
                (rotx*self.gridx - (1 - np.cos(rotx))*self.position)
                )
            roty_phase = np.exp(
                -1j*(2*np.pi/self.wavelength)*
                (roty*self.gridy - (1 - np.cos(roty))*self.position)
                )                
            #---------------------------------------------------
            # add vibration of the source
    
            offx = offx + np.sin(rotx) * self.position
            offy = offy + np.sin(roty) * self.position
            
            if offx > 0:
                loclx0 = _locate(self.xtick, self.xstart + offx)
                locrx0 = self.xcount - loclx0
                loclx1 = 0
                locrx1 = self.xcount - 2*loclx0
                
            elif offx <= 0:
                locrx0 = _locate(self.xtick, self.xend + offx)
                loclx0 = self.xcount - locrx0
                loclx1 = self.xcount - 2*locrx0
                locrx1 = self.xcount
            
            if offy > 0:
                locly0 = _locate(self.ytick, self.ystart + offy)
                locry0 = self.ycount - locly0
                locly1 = locly0
                locry1 = self.ycount - locly0
                
            elif offy <= 0:
                locry0 = _locate(self.ytick, self.yend + offy)
                locly0 = self.ycount - locry0
                locly1 = self.ycount - locry0
                locry1 = locry0
                
            for i in range(self.n):
            
                plane = deepcopy(self.plane)
                plane[:, loclx0 : locrx0] = (
                    (self.cmode[i] * 
                     rotx_phase * roty_phase)[:, loclx1 : locrx1]
                    )
                self.cmode[i] = plane
                

    # def expand(self, xcoor = None, ycoor = None):
        
    #     eplx = int(np.abs(xcoor[0] - self.xstart)/self.xpixel)
    #     eprx = int(np.abs(xcoor[1] - self.xend)/self.xpixel)
    #     eply = int(np.abs(ycoor[0] - self.ystart)/self.ypixel)
    #     epry = int(np.abs(ycoor[1] - self.yend)/self.ypixel)
        
    #     xcount = eplx + eprx + self.xcount
    #     ycount = eply + epry + self.ycount
        
    #     xstart = self.xstart - eplx * self.xpixel
    #     xend = eprx * self.xpixel + self.xend
    #     ystart = self.ystart - eply * self.ypixel
    #     yend = epry * self.ypixel + self.yend
        
    #     cmode = np.zeros((xcount, ycount), dtype = complex)
        
    #     for i in range(self.n):
    #         cmode[eplx : eplx + self.xcount, 
    #               eply : eply + self.ycount] = self.cmode[i]
    #         self.cmode[i] = np.copy(cmode)
    #         cmode = np.zeros((xcount, ycount), dtype = complex)
        
    #     self.xstart = xstart
    #     self.xend = xend
    #     self.ystart = ystart
    #     self.yend = yend
    #     self.xcount = xcount
    #     self.ycount = ycount
    #     self.xtick = np.linspace(xstart, xend, xcount)
    #     self.ytick = np.linspace(ystart, yend, ycount)
    #     self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
        
    #     self.mask = np.ones((self.xcount, self.ycount))

class screen(_optics):
    """
    Construct the class of screen.
    
    methods: 
    """
    
    def __init__(self,
                 optics = None, n = 0, location = 0, e = None):
        
        super().__init__(optic = optics, name = "screen", n_vector = n, 
                         position = location)
        
        self.cmode = [self.mask for i in range(self.n)]

        #---------------------------------------------------
        if e is None:
            error_phase = 1
        else:
            error_phase = np.exp(1j*e)
            # error_phase = e
            
        for i in range(self.n):
            self.cmode[i] = error_phase


class slit(_optics):
    """
    Construct the class of slit.
    
    methods: 
    """
    
    def __init__(self,
                 optics = None, n = 0, location = 0,
                 xc = None, yc = None, rd = None, sp = "b"):
        
        super().__init__(optic = optics, name = "slit", n_vector = n, 
                         position = location)
        
        super().add_mask(xcoor = xc, ycoor = yc, r = rd, s = sp)
        
        self.cmode = [self.mask for i in range(self.n)]
            
# class crl(_optics):
#     """
#     Construct the class of crl.
    
#     methods: 
#     """
    
#     def __init__(self, 
#                   optics = None, n = 0, location = 0,
#                   nlens = 0, delta = 2.216e-6, attenu = 89.5,
#                   rx = 0, ry = 0,
#                   t_lens = 0.5e-3, t_frame = 2e-3, g_aperture = 434e-6,
#                   offx = 0, offy = 0, rotx = 0, roty = 0, e = None):
    
        
#         super().__init__(optic = optics, name = "ideal lens", 
#                           n_vector = n, 
#                           position = location)
        
#         #---------------------------------------------------
#         # check geometry
        
#         if (abs(self.xstart - self.xend) < g_aperture or
#             abs(self.ystart - self.yend) < g_aperture):
#             return ValueError(
#                 "The size of optic plane is smaller than geometry aperture!"
#                 )
#         index = np.sqrt(self.gridx**2 + self.gridy**2)
        
#         #---------------------------------------------------
#         # the focus length of lens
        
#         self.focus_x = rx/(2*nlens*delta) if rx != 0 else 1e20
#         self.focus_y = ry/(2*nlens*delta) if ry != 0 else 1e20
        
#         #---------------------------------------------------
#         # the absorption plane of lens
        
#         # self.a_para = (t_frame - t_lens) / (g_aperture/2)**2
#         try:
#             self.a_para = 1 / (2*rx)
#         except:
#             self.a_para = 1 / (2*ry)
            
#         grid = np.sqrt(self.gridx**2 + self.gridy**2)
#         thickness = grid**2 * self.a_para + t_lens
#         thickness[index > g_aperture/2] = (
#             self.a_para * (g_aperture/2)**2 + t_lens
#             )
#         self.absorption = 1 - thickness * attenu
        
#         #---------------------------------------------------
#         # add vibration of source
        
        
#         self.lens_phase = np.zeros(
#             (self.xcount, self.ycount), dtype = np.complex128
#             )
#         lens_phase = np.exp(
#             1j*(2*np.pi/self.wavelength) *
#             ((self.gridx + offx)**2/(2*self.focus_x) + 
#               (self.gridy + offy)**2/(2*self.focus_y))
#             )
        
#         self.lens_phase[index > g_aperture/2] = 1
#         self.lens_phase[index < g_aperture/2] = (
#             lens_phase[index < g_aperture/2]
#             )
            
        
#         #---------------------------------------------------
#         # add rotation of source
        
#         rotx_phase = np.exp(
#             -1j*(2*np.pi/self.wavelength)*np.sin(rotx)*self.gridx
#             )
#         roty_phase = np.exp(
#             -1j*(2*np.pi/self.wavelength)*np.sin(roty)*self.gridy
#             ) 
        
#         #---------------------------------------------------
        
#         for i in range(self.n):
#             self.cmode[i] = self.lens_phase*rotx_phase*roty_phase
            
#         #---------------------------------------------------
#         if e is None:
#             error_phase = 1
#         else:
#             error_phase = np.exp(1j*e)
#             # error_phase = e
            
#         for i in range(self.n):
#             self.cmode[i] = (
#                 self.lens_phase*
#                 rotx_phase*roty_phase*
#                 self.absorption**0.5*
#                 error_phase
#                 )

class crl(_optics):
    """
    Construct the class of crl.
    
    methods: 
    """
    
    def __init__(self, 
                  optics = None, n = 0, location = 0,
                  nlens = 0, delta = 2.216e-6, 
                  rx = 0, ry = 0,
                  offx = 0, offy = 0, rotx = 0, roty = 0, e = None):
    
        
        super().__init__(optic = optics, name = "ideal lens", 
                          n_vector = n, 
                          position = location)
        
        #---------------------------------------------------
        # the focus length of lens
        
        self.focus_x = rx/(2*nlens*delta) if rx != 0 else 1e20
        self.focus_y = ry/(2*nlens*delta) if ry != 0 else 1e20
        
        #---------------------------------------------------
        # add vibration of source
        
        self.lens_phase = np.exp(
            1j*(2*np.pi/self.wavelength) *
            ((self.gridx + offx)**2/(2*self.focus_x) + 
             (self.gridy + offy)**2/(2*self.focus_y))
            )
        
        #---------------------------------------------------
        # add rotation of source
        
        rotx_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(rotx)*self.gridx
            )
        roty_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(roty)*self.gridy
            ) 
        
        #---------------------------------------------------
        
        for i in range(self.n):
            self.cmode[i] = self.lens_phase*rotx_phase*roty_phase
            
        #---------------------------------------------------
        if e is None:
            error_phase = 1
        else:
            error_phase = np.exp(1j*e)
            # error_phase = e
            
        for i in range(self.n):
            self.cmode[i] = (self.lens_phase*rotx_phase*roty_phase*error_phase)
            
# class KB_ideal(_optics):
#     """
#     Construct the class of ideal KB mirror.
    
#     methods: 
#     """
    
#     def __init__(self, 
#                  sr = None, optic = "crl", n = 0, location = 0,
#                  xfocus = 0, yfocus = 0, angle = 0,
#                  offx = 0, offy = 0, rotx = 0, roty = 0):
    
#         super().__init__(source = sr, name = optic, 
#                          n_vector = n, 
#                          position = location)
        
#         self.focus_x = xfocus
#         self.focus_y = yfocus
        
#         offx = np.sin(angle) * offx
#         offy = np.sin(angle) * offy
        
#         #---------------------------------------------------
#         # add vibration of source
        
#         self.lens_phase = np.exp(
#             1j*(2*np.pi/self.wavelength) *
#             ((self.gridx + offx)**2/(2*self.focus_x) + 
#              (self.gridy + offy)**2/(2*self.focus_y))
#             )
        
#         #---------------------------------------------------
#         # add rotation of source
        
#         # ideally the rocking angle of reflected light is double of rocking 
#         # angle of mirror
        
#         rotx = 2*rotx
#         roty = 2*roty
        
#         rotx_phase = np.exp(
#             -1j*(2*np.pi/self.wavelength)*np.sin(rotx)*self.gridx
#             )
#         roty_phase = np.exp(
#             -1j*(2*np.pi/self.wavelength)*np.sin(roty)*self.gridy
#             )    
#         #---------------------------------------------------
        
#         for i in range(self.n):
#             self.cmode.append(self.lens_phase*rotx_phase*roty_phase)

class VKB(_optics):
    """
    """
    def __init__(self, 
                 optics = None, n = 0, location = 0,
                 pfoucs = 0, qfocus = 0):
        
        super().__init__(optic = optics, name = "VKB", n_vector = n,
                         position = location)
        
        for i in range(self.n):
            
            self.cmode[i] = (
                np.exp(1j*(2*np.pi/self.wavelength) *
                       (np.sqrt((self.gridy + offset)**2 + pfocus**2) +
                        np.sqrt((self.gridy + offset)**2 + qfocus**2))) *
                np.exp(-1j*(2*np.pi/self.wavelength) *
                       np.sin(2*rot)*self.gridy)
                )

class HKB(_optics):
    """
    """
    def __init__(self,
                 optics = None, n = 0, location = 0,
                 pfocus = 0, qfocus = 0):
        
        super.__init__(optic = optics, name = "HKB", n_vector = n,
                       position = location)
        
        for i in range(self.n):
            
            self.cmode[i] = (
                np.exp(1j*(2*np.pi/self.wavelength) *
                       (np.sqrt((self.gridx + offset)**2 + pfocus**2) +
                        np.sqrt((self.gridx + offset)**2 + qfocus**2))) *
                np.exp(-1j*(2*np.pi/self.wavelength) *
                       np.sin(2*rot)*self.gridx)
                )
        
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
                 offset = 0,  rot = 0, e = None):
    
        super().__init__(optic = optics, name = "KB", 
                         n_vector = n, 
                         position = location)
        
        #---------------------------------------------------
        # add vibration of source
        
        # T0DO: if the source of vibration is earth, this asumpation is wrong.
        # offx = np.sin(angle) * offx
        # offy = np.sin(angle) * offy
        
        rotx_phase = 1
        roty_phase = 1
        
        if direction == 'h':
            rotx_phase = np.exp(-1j*(2*np.pi/self.wavelength) *
                                np.sin(2*rot)*self.gridx)
        elif direction == 'v':
            roty_phase = np.exp(-1j*(2*np.pi/self.wavelength) *
                                np.sin(2*rot)*self.gridy)
        
        #---------------------------------------------------
        # add rotation of source
        
        # ideally the rocking angle of reflected light is double of rocking 
        # angle of mirror
        
        if direction == 'h':
            self.lens_phase = np.exp(
                1j*(2*np.pi/self.wavelength) *
                (np.sqrt((self.gridx + offset)**2 + pfocus**2) +
                 np.sqrt((self.gridx + offset)**2 + qfocus**2))
                )
        
        elif direction == 'v':
            self.lens_phase = np.exp(
                1j*(2*np.pi/self.wavelength) *
                (np.sqrt((self.gridy + offset)**2 + pfocus**2) +
                 np.sqrt((self.gridy + offset)**2 + qfocus**2))
                )
            
        #---------------------------------------------------
        # add error phase
        
        if e is None:
            error_phase = 1
        else:
            if direction == 'v':
                error = np.zeros((self.xcount, self.ycount), dtype = float)
                for i in range(self.ycount): error[i, :] = e
            elif direction == 'h':
                error = np.zeros((self.xcount, self.ycount), dtype = float)
                for i in range(self.xcount): error[:, i] = e
                
            error_phase = np.exp(1j*e)
    
        for i in range(self.n):
            self.cmode[i] = self.lens_phase * rotx_phase * roty_phase


class AKB(_optics):
    """
    Construct the class of KB mirror. The effect of rocking and offset were
    considered.
    
    methods: 
    """
    
    def __init__(self, 
                 optics = None, direction = 'v', kind = 'ep',
                 n = 0, location = 0,
                 pfocus = 0, qfocus = 0,
                 afocus = 0, bfocus = 0,
                 length = 1, angle = 0,
                 offset = 0,  rot = 0, e = None):
    
        super().__init__(optic = optics, name = "AKB", 
                         n_vector = n, 
                         position = location)
        
        #---------------------------------------------------
        # add vibration of source
        
        # T0DO: if the source of vibration is earth, this asumpation is wrong.
        # offx = np.sin(angle) * offx
        # offy = np.sin(angle) * offy
        
        rotx_phase = 1
        roty_phase = 1
        
        if direction == 'h':
            rotx_phase = np.exp(-1j*(2*np.pi/self.wavelength) *
                                np.sin(2*rot)*self.gridx)
        elif direction == 'v':
            roty_phase = np.exp(-1j*(2*np.pi/self.wavelength) *
                                np.sin(2*rot)*self.gridy)
        
        #---------------------------------------------------
        # add rotation of source
        
        # ideally the rocking angle of reflected light is double of rocking 
        # angle of mirror
        
        if direction == 'h':
            
            if kind == 'ep':
                
                self.lens_phase = np.exp(
                    1j*(2*np.pi/self.wavelength) *
                    (np.sqrt((self.gridx + offset)**2 + pfocus**2) +
                     np.sqrt((self.gridx + offset)**2 + qfocus**2))
                    )
            
            elif kind == 'hb':
                
                self.lens_phase = np.exp(
                    1j*(2*np.pi/self.wavelength) *
                    (np.sqrt((self.gridx + offset)**2 + afocus**2) -
                     np.sqrt((self.gridx + offset)**2 + bfocus**2))
                    )
        
        elif direction == 'v':
            
            if kind == 'ep':
            
                self.lens_phase = np.exp(
                    1j*(2*np.pi/self.wavelength) *
                    (np.sqrt((self.gridy + offset)**2 + pfocus**2) +
                     np.sqrt((self.gridy + offset)**2 + qfocus**2))
                    )
            
            elif kind == 'hb':
            
                self.lens_phase = np.exp(
                    1j*(2*np.pi/self.wavelength) *
                    (np.sqrt((self.gridy + offset)**2 + afocus**2) -
                     np.sqrt((self.gridy + offset)**2 + bfocus**2))
                    )
        
        #---------------------------------------------------
        # add error of the mirror
        
        error = np.ones((self.ycount, self.xcount), dtype = np.complex128)
        
        if e is not None:
            if direction == 'h':
                for i in range(self.ycount): error[i, :] = np.exp(1j*e)
            elif direction == 'v':
                for i in range(self.xcount): error[:, i] = np.exp(1j*e)
        
        #--------------------------------------------------
        # construct phase and error
        
        for i in range(self.n):
            self.cmode[i] = self.lens_phase * error
            
            

class ideal_lens(_optics):
    """
    Construct the class of ideal lens
    
    methods
    """
    
    def __init__(self, optics = None, n = 0, location = 0,
                 xfocus = 0, yfocus = 0, offx = 0, offy = 0, 
                 rotx = 0, roty = 0, e = None):
        
        super().__init__(optic = optics, name = "ideal lens", 
                         n_vector = n, 
                         position = location)
        
        self.focus_x = xfocus
        self.focus_y = yfocus
    
        #---------------------------------------------------
        # add vibration of source
        
        self.lens_phase = np.exp(
            1j*(2*np.pi/self.wavelength) *
            ((self.gridx + offx)**2/(2*self.focus_x) + 
             (self.gridy + offy)**2/(2*self.focus_y))
            )
        
        #---------------------------------------------------
        # add rotation of source
        
        rotx_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(rotx)*self.gridx
            )
        roty_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(roty)*self.gridy
            )    
        #---------------------------------------------------
        if e is None:
            error_phase = 1
        else:
            error_phase = np.exp(1j*e)
            # error_phase = e
            
        for i in range(self.n):
            self.cmode[i] = self.lens_phase*rotx_phase*roty_phase*error_phase
            
#-----------------------------------------------------------------------------#

class rot_DCM(_optics):
    
    """
    The rocking of DCM
    """
    
    def __init__(self, optics = None, n = 0, location = 0,
                 rotx = 0,roty = 0, e = None):
        
        super().__init__(optic = optics, name = "ideal_lens",
                         n_vector = n,
                         position = location)
        
        rotx_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(rotx)*self.gridx
            )
        roty_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(roty)*self.gridy
            )
        
        for i in range(self.n):
            self.cmode[i] = rotx_phase*roty_phase
            
            