#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
configure: The process of source construction.

Functions: None.
           
Classes  : None.
"""

#-----------------------------------------------------------------------------#
# library

from cat._source import _source
from cat._source import _decomposition

#-----------------------------------------------------------------------------#
# parameters

undulator = {
        "period_length":        0.0183,
        "period_number":        77.0,
        "n_hormonic":           1,
        "hormonic_energy":      17050,
        "direction":            "v",
        "symmetry_v":           -1,
        "symmetry_h":           0
        }

electron_beam = {
        "n_electron":           90000,
        "current":              0.2,
        "energy":               6.0,
        "energy_spread":        0.93e-03,
        # "energy_spread":        0.0,
        "sigma_x0":             30.18e-06,
        "sigma_xd":             3.64e-06,
        "sigma_y0":             4.37e-06,
        "sigma_yd":             1.37e-06
        }

screen = {
        "xstart":               -0.00025,
        "xfin":                 0.00025,
        "nx":                   100,
        "ystart":               -0.00025,
        "yfin":                 0.00025,
        "ny":                   100,
        "screen":               10.0,
        "n_vector":             500
        }

#-----------------------------------------------------------------------------#

_source._cal_wfrs(
        undulator, electron_beam, screen, 
        file_name = "ebs_srw.h5", method = 'srw'
        )
_decomposition._multi_layer_svd_exceed_0(
        50000, int(100), int(100), 500, file_name = "ebs_srw.h5"
        )