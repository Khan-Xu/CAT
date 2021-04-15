#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS Hard X-ray Scattering Beamline (B4)"
__date__     = "Date : 05.01.2021"

"""
B4 XPCS optic setup.
"""

#-----------------------------------------------------------------------------#
import sys
sys.path.append(r'G:\CAT_beta_1.0')

#-----------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt

from cat.optics import source, ideal_lens, screen
from cat.propagate import propagate_s
from cat import tool

#------------------------------------------------------------------------------
# parameters

# the number of coheret modes to propagate
ncount = 30

#------------------------------------------------------------------------------

#---------------------------------------------------
# source 
sr0 = source(file_name = "B4_test.h5", name = "source", n_vector = ncount)

#remap the coherent mode
sr0.remap(0.5e-6, 0.5e-6)

#---------------------------------------------------
# white beam slit
wbs = screen(optics = sr0, n = ncount, location = 40)
propagate_s(sr0, wbs)

#---------------------------------------------------
# crl
crl = ideal_lens(optics = wbs, n = ncount, location = 45,
                 xfocus = 19.138, yfocus = 19.138)
propagate_s(wbs, crl)
crl.add_mask(xcoor = [-2.15e-4, 2.15e-4], ycoor = [-2.15e-4, 2.15e-4])

#---------------------------------------------------
# focus
focus = screen(optics = crl, n = ncount, location = 78.3)
propagate_s(crl, focus)

# apply secondary source slit

# delta: adjust the shift of coherent mode. (FFT)
delta = 1.21629e-6

focus.slit(xcoor = [-5e-5, 5e-5], ycoor = [-5e-5, 5e-5])
focus.add_mask(xcoor = [-2.5e-5 + delta, 2.5e-5 + delta + 0.5e-6], 
               ycoor = [-2e-5, 2e-5])

# If remap is applied, the boundary should be mask.
focus.remap(0.5e-6, 0.5e-6)

focus.add_mask(xcoor = [-2.0e-5 + delta, 2.0e-5 + delta + 1.0e-6], 
               ycoor = [-2e-5, 2e-5])

# save focus with slit
focus.name = "xpcs_focus_withslit_20um"
focus.save()

#---------------------------------------------------
# sample slit
ss = screen(optics = focus, n = ncount, location = 80.5)
propagate_s(focus, ss)
ss.add_mask(xcoor = [-1e-5 + 1.2574e-6, 1e-5 + 1.2574e-6 + 0.4e-6], 
            ycoor = [-1e-5 + 0.25e-6,   1e-5 + 0.25e-6   + 0.4e-6])

# If remap is applied, the boundary should be mask.
ss.remap(0.05e-6, 0.05e-6)

ss.add_mask(xcoor = [-0.7e-5 + 1.2574e-6, 0.7e-5 + 1.2574e-6 + 0.4e-6], 
            ycoor = [-0.7e-5 + 0.25e-6,   0.7e-5 + 0.25e-6   + 0.4e-6])

ss.name = "sampleslit_slit_size_14um"
ss.save()

#---------------------------------------------------
# sample
sample = screen(optics = ss, n = ncount, location = 81.0)
propagate_s(ss, sample)

sample.slit(xcoor = [-2e-5, 2e-5], ycoor = [-2e-5, 2e-5])
# sample.name = "xpcs_slit_distance_0.05m_slit_size_5um"
# sample.save()
#------------------------------------------------------------------------------

