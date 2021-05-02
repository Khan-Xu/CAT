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

#------------------------------------------------------------------------------
# parameters

_ncount = 30
x_pixel = 0.039e-6
y_pixel = 0.039e-6

#------------------------------------------------------------------------------
# beamline

sr0 = source2(
    file_name = 'b4_srw2_12400.h5', name = 'source', n_vector = _ncount
    )

sr0.expand(xcoor = [-4e-4, 4e-4], ycoor = [-4e-4, 4e-4])
sr0.remap(0.5e-6, 0.5e-6)

crl = ideal_lens(
    optics = sr0, n = _ncount, location = 40, 
    xfocus = 16.91, yfocus = 16.91
    )

ssa = screen(optics = sr0, n = _ncount, location = 69.3)

#------------------------------------------------------------------------------
# proapgation
    
propagate_s(sr0, crl)
propagate_s(crl, ssa)

#------------------------------------------------
ix = np.sum(np.abs(ssa.cmode[0]), 0)
iy = np.sum(np.abs(ssa.cmode[0]), 1)

x_loc = np.argmin(np.abs(ix - np.max(ix)))
y_loc = np.argmin(np.abs(iy - np.max(iy)))

center_x = ssa.xtick[x_loc]
center_y = ssa.ytick[y_loc]

ssa.add_mask(
    xcoor = [-10.5e-6 + center_x, 10.5e-6 + center_x],
    ycoor = [-10.5e-6 + center_y, 10.5e-6 + center_y]
    )

#--------------------------------------------------

scr = screen(optics = ssa, n = _ncount, location = 72.0)
propagate_s(ssa, scr)

#------------------------------------------------------------------------------
# plot sample plane

scr.slit(xcoor = [-20e-6, 20e-6], ycoor = [-20e-6, 20e-6])
scr.remap(x_pixel, y_pixel)

ix = np.sum(np.abs(scr.cmode[0]), 0)
iy = np.sum(np.abs(scr.cmode[0]), 1)

x_loc = np.argmin(np.abs(ix - np.max(ix)))
y_loc = np.argmin(np.abs(iy - np.max(iy)))

center_x = scr.xtick[x_loc]
center_y = scr.ytick[y_loc]

scr.slit(
    xcoor = [-5e-6 + center_x, 5e-6 + center_x],
    ycoor = [-5e-6 + center_y, 5e-6 + center_y]
    )

scr.xtick = scr.xtick - center_x
scr.ytick = scr.ytick - center_y

#------------------------------------------------------------
# plot coherent modes

scr._svd()
tool.plot_optic(scr, t = 'mode', n = (1, 3))




