#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"

#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

from cat.utils._optics import _locate
from cat.optics        import source, source2, ideal_lens, screen, slit
from cat.propagate     import propagate_s
from cat               import tool
from cat               import phase 

#------------------------------------------------------------------------------
# parameters

_ncount = 30

#------------------------------------------------------------------------------
# beamline

sr0 = source2(
    file_name = 'b4_srw2_12400.h5', name = 'source', n_vector = _ncount
    )

sr0.remap(0.5e-6, 0.5e-6)

crl = ideal_lens(
        optics = sr0, n = _ncount, location = 40, 
        xfocus = 16.91, yfocus = 16.91
        )

focus = screen(optics = sr0, n = _ncount, location = 69.3)

#------------------------------------------------------------------------------
# proapgation
    
propagate_s(sr0, crl)
propagate_s(crl, focus)

#------------------------------------------------------------------------------
# plot focus

center_x = 3.19
focus.xtick = focus.xtick - center_x*1e-6
focus.slit(xcoor = [-3e-5, 3e-5], ycoor = [-3e-5, 3e-5])
focus.cal_csd()

tool.plot_optic(focus, t = 'intensity')
tool.plot_optic(focus, t = 'csd')



