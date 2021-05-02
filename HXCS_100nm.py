#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
100 nm focus beamline layout of b4.
"""

#-----------------------------------------------------------------------------#
# modules

import sys
sys.path.append("..")
import gc as garbage_collect

import numpy as np
import matplotlib.pyplot as plt

from cat.utils._optics import _locate
from cat.optics import source, source2, ideal_lens, KB, screen, slit, AKB
from cat.propagate import propagate_s, propagate_k, propagate_czt
from cat import tool

#------------------------------------------------------------------------------
# parameters

_ncount = 1

# the loop of different modes
_vectors = range(_ncount)

# parameters for vkb
_vkb_ep_location = 79.620 - 0.158/2
_vkb_ep_pfocus = 79.620 - 0.158/2
_vkb_ep_qfocus = 0.76292

_vkb_hb_location = 79.620 + 0.158/2
_vkb_hb_afocus = 0.301
_vkb_hb_bfocus = 0.76292 - 0.158

# parameters for hkb
_hkb_ep_location = 79.870 - 0.079/2
_hkb_ep_pfocus = 79.870 - 0.079/2
_hkb_ep_qfocus = 0.2603

_hkb_hb_location = 79.870 + 0.079/2
_hkb_hb_afocus = 0.0905
_hkb_hb_bfocus = 0.2603 - 0.079

#------------------------------------------------------------------------------
# function

#------------------------------------------------------------------------------
# beamline functions

e = 0

akb = 0

offsetx = 0
offrotx = 0

# def beamline_saCDI(offsetx = 0, offrotx = 0, e = 0):

#---------------------------------------------------
# source - 20m
    
sr0 = source(
    file_name = "B4.h5", name = "source", 
    n_vector = _ncount, 
    offx = offsetx, rotx = offrotx
    )
sc0 = screen(optics = sr0, n = _ncount, location = 20)
propagate_s(sr0, sc0)
sc0.expand(xcoor = [-8e-4, 8e-4], ycoor = [-8e-4, 8e-4])
sc0.remap(1e-6, 1e-6)

#---------------------------------------------------
# DCM slit

dcm = screen(optics = sc0, n = _ncount, location = 40)
propagate_s(sc0, dcm)

# add phase error
dcm.slit(xcoor = [-3e-4, 3e-4], ycoor = [-3e-4, 3e-4])
dcm.expand(xcoor = [-8e-4, 8e-4], ycoor = [-8e-4, 8e-4])
dcm.add_mask(xcoor = [-3e-4, 3e-4], ycoor = [-3e-4, 3e-4])

#---------------------------------------------------
# KB slit

sc2 = screen(optics = dcm, n = _ncount, location = _vkb_ep_location - 0.158)
propagate_s(dcm, sc2)

hslit = 1.597e-4
vslit = 3.981e-4

s = -4.78e-6

sc2.slit(
    xcoor = [-vslit/2 + s, vslit/2 + s], 
    ycoor = [-vslit/2 - 1e-5, vslit/2 + 1e-5]
    )
sc2.add_mask(
    xcoor = [-hslit/2 + s, hslit/2 + s], 
    ycoor = [-vslit/2, vslit/2]
    )
sc2.remap(0.08e-6, 0.08e-6)

#---------------------------------------------------
# KB focusw

# propagate to vkb

vkb_ep = AKB(
    optics = sc2, direction = 'v', kind = 'ep', n = _ncount, 
    location = _vkb_ep_location, 
    pfocus = _vkb_ep_pfocus, qfocus = _vkb_ep_qfocus
    )
propagate_s(sc2, vkb_ep, t = 'asm')

vkb_hb = AKB(
    optics = vkb_ep, direction = 'v', kind = 'hb', n = _ncount, 
    location = _vkb_hb_location, 
    afocus = _vkb_hb_afocus, bfocus = _vkb_hb_bfocus
    )
propagate_s(vkb_ep, vkb_hb, t = 'asm')

# propagate to hkb

hkb_ep = AKB(
    optics = vkb_hb, direction = 'h', kind = 'ep', n = _ncount, 
    location = _hkb_ep_location, 
    pfocus = _hkb_ep_pfocus, qfocus = _hkb_ep_qfocus
    )
propagate_s(vkb_hb, hkb_ep, t = 'asm')

hkb_hb = AKB(
    optics = hkb_ep, direction = 'h', kind = 'hb', n = _ncount, 
    location = _hkb_hb_location, 
    afocus = _hkb_hb_afocus, bfocus = _hkb_hb_bfocus
    )
propagate_s(hkb_ep, hkb_hb, t = 'asm')

#---------------------------------------------------
# propagate to foucs palne
    
sc4 = screen(optics = hkb_hb, n = _ncount, location = 80.0)
sc4.slit(
    xcoor = [-0.55e-6, 0.25e-6 + 1.5e-8], 
    ycoor = [-0.7e-6, 0.1e-6 + 1.5e-8], 
    t = 1
    )
sc4.remap(1.5e-8, 1.5e-8)

propagate_czt(hkb_hb, sc4)

tool.plot_optic(sc4, t = 'intensity')
