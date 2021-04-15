#-----------------------------------------------------------------------------#
# Copyright (c) 2021 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS HXS (B4) xuhan@ihep.ac.cn"
__date__     = "Date : 04.01.2021"
__version__  = "beta-1.0"


"""
configure: The process of source construction.

Functions: None.
           
Classes  : None.
"""

#-----------------------------------------------------------------------------#
# library

from cat import source

#-----------------------------------------------------------------------------#
# parameters

#-------------------------------------------------------------
"""
the parameters of undulator

Args: period_length   - the length of one period of undulator (metre).
      period_number   - the number of the periods of undulator.
      n_hormonic      - the hormonic number.
      harmonic_energy - the harmonic energy.
      direction       - the drection of magnetic field. 
                        "v" for vertical, 
                        "h" for horizontal, 
                        "b" for both.
      symmetry_v      - "-1" for asymmetry, "0" for none, "1" for symmetry.
      symmetry_h      - "-1" for asymmetry, "0" for none, "1" for symmetry.
"""
undulator = {
        "period_length":        0.0199,
        "period_number":        201.0,
        "n_hormonic":           1,
        "hormonic_energy":      12398.4,
        "direction":            "v",
        "symmetry_v":           -1,
        "symmetry_h":           0
        }
#-------------------------------------------------------------
"""
the parameters of electron beam.

Args: n_electron    - the number of electrons for monte carlo.
      current       - the current of electron beam (A).
      energy        - the energy of electron beam (GeV).
      energy_spread - the energy spread of electron beam.
      sigma_x0      - the gaussian sigma of electron horizontal 
                      position (metre).
      sigma_xd      - the gaussian sigma of electron horizontal 
                      divergence (rad).
      sigma_y0      - the gaussian sigma of electron vertical 
                      position (metre).
      sigma_yd      - the gaussian sigma of electron vertical 
                      divergence (rad).
"""
electron_beam = {
        "n_electron":           1000,
        "current":              0.2,
        "energy":               6.0,
        "energy_spread":        2e-04,
        "sigma_x0":             9.334e-06,
        "sigma_xd":             2.438e-06,
        "sigma_y0":             3.331e-06,
        "sigma_yd":             1.275e-06
        }
#-------------------------------------------------------------
"""
the parameters of screen.

Args: xstart   - the start horizontal postion of screen (metre).
      xfin     - the end horizontal position of screen (metre).
      nx       - the number of pixel along horizontal direction.
      ystart   - the start vertical position of screen (metre).
      yfin     - the end vertical position of scree (metre).
      ny:      - the number of pixel along vertical direction.
      screen   - the location of the recived plane.
      n_vertor - the number of the decomposed coherent modes.
      
"""
screen = {
        "xstart":               -0.0003,
        "xfin":                 0.0003,
        "nx":                   100,
        "ystart":               -0.0003,
        "yfin":                 0.0003,
        "ny":                   100,
        "screen":               20.0,
        "n_vector":             200
        }
#------------------------------------------------------------------------------
"""
source_name - the name of the source.
"""

source_name = "HEPS_B4"

#-----------------------------------------------------------------------------#

source.multi_electron(undulator, 
                      electron_beam, 
                      screen, 
                      file = source_name + ".h5")


