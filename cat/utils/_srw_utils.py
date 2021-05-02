#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_srw_utils: The construction of beam source base on srwpy.

Functions: None.
           
Classes  : _srw_electron_beam    - construct e_beam base on monte carlo method.
           _undulator            - undulator parameters setting.
           _propagete_wave_front - propagate wavafront from source to screen.
"""

#-----------------------------------------------------------------------------#
# library

import numpy           as np
import scipy.constants as codata

from cat.utils     import _constant
from array         import *
from math          import *
from copy          import deepcopy
from srwpy.srwlib  import *

#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function

#-----------------------------------------------------------------------------#
# class

class _srw_electron_beam(object):
    
    """
    ---------------------------------------------------------------------------
    description: construct e_beam base on monte carlo method.
    
    methods: monte_carlo       - monte carlo paramters of electron beam.
             after_monte_carlo - electron beam after monte carlo.
    ---------------------------------------------------------------------------
    """
    
    def __init__(self, electron_beam, n_period, period_length):
        
        # the initail postion of electron beam
        self.initial_z = -period_length * (n_period + 8)/2
        gamma = electron_beam["energy"]/_constant._E_r
        
        self.part_beam = SRWLPartBeam()
        
        self.part_beam.Iavg = electron_beam['current']
        self.part_beam.partStatMom1.z = self.initial_z
        self.part_beam.partStatMom1.gamma = gamma
        
        # energy spread, sigma_x, sigma_xp, sigma_y, sigma_yp
        self.part_beam.arStatMom2[10] = electron_beam['energy_spread']**2
        self.part_beam.arStatMom2[0] = electron_beam['sigma_x0']**2
        self.part_beam.arStatMom2[2] = electron_beam['sigma_xd']**2
        self.part_beam.arStatMom2[3] = electron_beam['sigma_y0']**2
        self.part_beam.arStatMom2[5] = electron_beam['sigma_yd']**2
        
        self.mc = dict()
    
    def monte_carlo(self):
        
        """
        -----------------------------------------------------------------------
        description: calculate monte carlo paramters of electron beam.
        
        args: none.
        
        return: none.
        -----------------------------------------------------------------------
        """
    
        _part_beam = deepcopy(self.part_beam)
        # The electron beam parameters
        self.mc["mult_x"] = (
                  0.5 / 
                  (_part_beam.arStatMom2[0] * _part_beam.arStatMom2[2] - 
                   _part_beam.arStatMom2[1] * _part_beam.arStatMom2[1])
                  )
        self.mc["bx"] = _part_beam.arStatMom2[0] * self.mc["mult_x"]
        self.mc["gx"] = _part_beam.arStatMom2[2] * self.mc["mult_x"]
        self.mc["ax"] = _part_beam.arStatMom2[1] * self.mc["mult_x"]
        self.mc["sigma_px"] = 1/sqrt(2*self.mc["gx"])
        self.mc["sigma_qx"] = sqrt(
            self.mc["gx"] /
            (2*(self.mc["bx"]*self.mc["gx"] - self.mc["ax"]*self.mc["gx"]))
            )
        
        self.mc["mult_y"] = (
                  0.5 / 
                  (_part_beam.arStatMom2[3] * _part_beam.arStatMom2[5] - 
                   _part_beam.arStatMom2[4] * _part_beam.arStatMom2[4])
                  )
        self.mc["by"] = _part_beam.arStatMom2[3] * self.mc["mult_y"]
        self.mc["gy"] = _part_beam.arStatMom2[5] * self.mc["mult_y"]
        self.mc["ay"] = _part_beam.arStatMom2[4] * self.mc["mult_y"]
        self.mc["sigma_py"] = 1/sqrt(2*self.mc["gy"])
        self.mc["sigma_qy"] = sqrt(
            self.mc["gy"] /
            (2*(self.mc["by"]*self.mc["gy"] - self.mc["ay"]*self.mc["gy"]))
            )
        
    def after_monte_carlo(self, rand_array, period_length, 
                          k_vertical, k_horizontal):
        
        """
        -----------------------------------------------------------------------
        description: calculate electron beam with monte carlo process.
        
        args: rand_array    - gaussian random number for sx, sxp, sy, syp.
              period_length - undualtor period length.
              k_vertical    - k value of undulator vertical axis.
              k_horizontal  - k value of undulator horizontal axis.
              
        return: _part_beam       - electron particle beam.
                wavelength       - 
                resonance_energy - 
        -----------------------------------------------------------------------
        """
        
        _part_beam = deepcopy(self.part_beam)
        
        # monte carlo process
        auxp_xp = self.mc["sigma_qx"] * rand_array[0]
        auxp_yp = self.mc["sigma_qy"] * rand_array[2]
        auxp_x0 = (self.mc["sigma_px"] * rand_array[1] + 
                   self.mc["ax"] * 
                   auxp_xp / self.mc["gx"])
        auxp_y0 = (self.mc["sigma_py"] * rand_array[3] + 
                   self.mc["ay"] * 
                   auxp_yp / self.mc["gy"])
        
        _part_beam.partStatMom1.x  = _part_beam.partStatMom1.x  + auxp_x0
        _part_beam.partStatMom1.y  = _part_beam.partStatMom1.y  + auxp_y0
        
        _part_beam.partStatMom1.xp = _part_beam.partStatMom1.xp + auxp_xp
        _part_beam.partStatMom1.yp = _part_beam.partStatMom1.yp + auxp_yp
        
        _part_beam.partStatMom1.gamma = (
                    self.part_beam.partStatMom1.gamma * 
                    (1 + sqrt(self.part_beam.arStatMom2[10]) * 
                    rand_array[4])
                    )
        gamma = deepcopy(_part_beam.partStatMom1.gamma)
        wavelength = (
            (period_length/(2.0*gamma**2))*
            (1 + k_vertical**2/2.0 + k_horizontal**2/2.0)
            )
        resonance_energy = _constant._Resonance_Factor / wavelength
        
        return _part_beam, wavelength, resonance_energy

#-------------------------------------------------------------
       
class _undulator(object):
    
    """
    ---------------------------------------------------------------------------
    descriptionsetting unduatlor paramters.
    
    methods: magnetic_structure - magnetic structure of undulator.
             wavelength         - calcualte wavelength.
             cal_k              - calcualte k value.
    ---------------------------------------------------------------------------
    """
    
    def __init__(self, undulator):
        
        self.period_length = undulator["period_length"]
        self.n_period = undulator["period_number"]
        
        self.n_hormonic = undulator["n_hormonic"]
        self.hormonic_energy = undulator["hormonic_energy"]
        self.magnetic_field_h = 0
        self.magnetic_field_v = 0
        
        self.wavelength = 0
        self.k = 0
        self.direction = undulator["direction"]
        self.symmetry = [undulator["symmetry_v"], undulator["symmetry_h"]]
        self.k_horizontal = 0
        self.k_vertical = 0
    
    def magnetic_structure(self):
        
        """
        -----------------------------------------------------------------------
        description: magnetic structure of undulator..
        
        args: none.
        
        return: magnetic_field_container - magnetic field structure of 
                                           undualtor.
        -----------------------------------------------------------------------
        """
        
        mult = (_constant._ElCh / 
                (2*np.pi * _constant._ElMass_kg * _constant._LightSp))
        b_vertical = self.k_vertical / (mult * self.period_length)
        b_horizontal = self.k_horizontal / (mult * self.period_length)
        magnetic_fields = []
        
        if self.direction == 'v':
            magnetic_fields.append(
                SRWLMagFldH(_h_or_v = self.direction, 
                            _B = b_vertical,
                            _s = self.symmetry[0])
                )
        elif self.direction == 'h':
            magnetic_fields.append(
                SRWLMagFldH(_h_or_v = self.direction,
                            _B = b_horizontal,
                            _s = self.symmetry[1])
                )
        elif self.direction == 'b':
            magnetic_fields.append(
                SRWLMagFldH(_h_or_v = 'v', _B = b_vertical, 
                            _s = self.symmetry[0])
                )
            magnetic_fields.append(
                SRWLMagFldH(_h_or_v = 'h', _B = b_horizontal, 
                            _s = self.symmetry[1])
                )
        
        magnetic_structure = SRWLMagFldU(
            _arHarm = magnetic_fields,
            _per = self.period_length,
            _nPer = self.n_period
            )
        magnetic_field_container = SRWLMagFldC(
            _arMagFld = [magnetic_structure],
            _arXc = array('d', [0.0]),
            _arYc = array('d', [0.0]),
            _arZc = array('d', [0.0])
            )
        
        return magnetic_field_container
        
    def wave_length(self):
        
        """
        -----------------------------------------------------------------------
        description: calcualte wavelength.
        
        args: none.
        
        return: none.
        -----------------------------------------------------------------------
        """
        
        self.wavelength = (codata.c*codata.h/codata.e) / self.hormonic_energy
        
    def cal_k(self, electron_beam_energy = 6):
        
        """
        -----------------------------------------------------------------------
        description: calcualte k value.
        
        args: None.
        
        return: None
        -----------------------------------------------------------------------
        """
        
        gamma = electron_beam_energy/_constant._E_r
        self.k = np.sqrt(
            2 * (((2 *self.wavelength * gamma**2) / self.period_length) - 1)
            )
        
        if self.direction == 'b':
            
            self.k_horizontal = self.k / np.sqrt(2)
            self.k_vertical = self.k / np.sqrt(2)  
        
        elif self.direction == 'v':  
            
            self.k_horizontal = 0
            self.k_vertical = np.copy(self.k)      
        
        elif self.direction == 'h':  
            
            self.k_horizontal = np.copy(self.k)
            self.k_vertical = 0

#-------------------------------------------------------------
            
class _propagate_wave_front(object):
    
    """
    ---------------------------------------------------------------------------
    description: propagate wavefront from source to screen.
    
    methods: _cal_wave_front - calcualte wave front from source to screen.
    ---------------------------------------------------------------------------
    """
    
    def __init__(self, wave_front, resonance_energy):
        
        self.mesh = SRWLRadMesh(
                _eStart = resonance_energy,
                _eFin = resonance_energy,
                _ne = 1,
                _xStart = wave_front['xstart'],
                _xFin = wave_front['xfin'],
                _nx = wave_front['nx'],
                _yStart = wave_front['ystart'],
                _yFin = wave_front['yfin'],
                _ny = wave_front['ny'],
                _zStart = wave_front['screen']
                )
        self.wfr = SRWLWfr()
        
    def _cal_wave_front(self, part_beam, magnetic_container):
        
        """
        -----------------------------------------------------------------------
        description: calcualte wavefront at the screen.
        
        args: none.
        
        return: none
        -----------------------------------------------------------------------
        """
        
        self.wfr.allocate(self.mesh.ne, self.mesh.nx, self.mesh.ny)
        self.wfr.mesh = deepcopy(self.mesh)
        self.wfr.partBeam = deepcopy(part_beam)
        srwl.CalcElecFieldSR(
            self.wfr, 0, magnetic_container, [1, 0.01, 0.0, 0.0, 50000, 1, 0.0]
            )
        
#-----------------------------------------------------------------------------#
