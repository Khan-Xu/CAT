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

from cat.utils import _source
from cat.utils import _decomposition

#-----------------------------------------------------------------------------#
# parameters

undulator = {
        "period_length":        0.0199,
        "period_number":        201.0,
        "n_hormonic":           1,
        "hormonic_energy":      12400,
        "direction":            "v",
        "symmetry_v":           -1,
        "symmetry_h":           0
        }

electron_beam = {
        "n_electron":           80000,
        "current":              0.2,
        "energy":               6.0,
        "energy_spread":        1.06e-03,
        # "energy_spread":        0.0,
        "sigma_x0":             9.334e-06,
        "sigma_xd":             3.331e-06,
        "sigma_y0":             2.438e-06,
        "sigma_yd":             1.275e-06
        }

screen = {
        "xstart":               -0.0003,
        "xfin":                 0.0003,
        "nx":                   150,
        "ystart":               -0.0003,
        "yfin":                 0.0003,
        "ny":                   150,
        "screen":               20.0,
        "n_vector":             300
        }

#-----------------------------------------------------------------------------#

_source._cal_wfrs(
        undulator, electron_beam, screen, 
        file_name = "b4_srw2_12400.h5", method = 'srw'
        )
_decomposition._multi_layer_svd_exceed_0(
        electron_beam['n_electron'], 
        int(screen['nx']), int(screen['ny']), int(screen['n_vector']),
        file_name="b4_srw2_12400.h5"
        )



