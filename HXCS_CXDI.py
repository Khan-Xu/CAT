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

x_pixel = 0.039e-6
y_pixel = 0.039e-6

supp_x_dim = 800
supp_y_dim = 800

samp_x_dim = 256
samp_y_dim = 256

#------------------------------------------------------------------------------
# beamline

iset = 2
CF = list()
intensitys = list()
flux_ratio = list()

islit = 7    

sr0 = source2(
    file_name = 'b4_srw2_12400.h5', name = 'source', n_vector = _ncount
    )
sr0.expand(xcoor = [-4e-4, 4e-4], ycoor = [-4e-4, 4e-4])
sr0.remap(0.5e-6, 0.5e-6)
sr0.cal_i()

crl = ideal_lens(
    optics = sr0, n = _ncount, location = 40, 
    xfocus = 16.91, yfocus = 16.91
    )

ssa = screen(optics = sr0, n = _ncount, location = 69.3)

#--------------------------------------------------------------------------
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
    xcoor = [-10.5e-6 + center_x + 2*islit*0.3e-6, 
              10.5e-6 + center_x - 2*islit*0.3e-6],
    ycoor = [-10.5e-6 + center_y + 2*islit*0.3e-6, 
              10.5e-6 + center_y - 2*islit*0.3e-6]
    )

#--------------------------------------------------

scr = screen(optics = ssa, n = _ncount, location = 72.0)
propagate_s(ssa, scr)

#----------------------------------------------------------------------
# plot sample plane

scr.slit(xcoor = [-5e-5, 5e-5], ycoor = [-5e-5, 5e-5])
# scr.remap(x_pixel, y_pixel)
scr.remap(0.1e-6, 0.1e-6)

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
scr.remap(x_pixel, y_pixel)

#----------------------------------------------------------------------
# sample

star = phase.siemens_star()
sample = np.zeros((supp_x_dim, supp_y_dim), dtype = np.complex128)
support = np.zeros((supp_x_dim, supp_y_dim), dtype = int)
support[
    400 - int(samp_x_dim/2) : 400 + int(samp_x_dim/2), 
    400 - int(samp_y_dim/2) : 400 + int(samp_y_dim/2)
    ] = 1
amp = star[2144 : 2400, 2144 : 2400]
sample[
    400 - int(samp_x_dim/2) : 400 + int(samp_x_dim/2), 
    400 - int(samp_y_dim/2) : 400 + int(samp_y_dim/2)
    ].real = amp

sample_center = sample[
    400 - int(samp_x_dim/2) : 400 + int(samp_x_dim/2), 
    400 - int(samp_y_dim/2) : 400 + int(samp_y_dim/2)
    ]

#----------------------------------------------------------------------
# diffraction

intensity = np.zeros((800, 800), dtype = float)
Sample = np.copy(sample)

scr._svd()
scr.n = 6

for i in range(scr.n):
    
    Sample[
        400 - int(samp_x_dim/2) : 400 + int(samp_x_dim/2),
        400 - int(samp_y_dim/2) : 400 + int(samp_y_dim/2)
        ] = (
            sample_center * 
            np.sqrt(scr.ratio[i]) * 
            scr.cmode[i][0 : samp_x_dim, 0 : samp_y_dim]
            ) 
    
    intensity += np.abs(sp.fft.fftshift(sp.fft.fft2(Sample)))**2
    
#----------------------------------------------------------------------
# reconstruction

print("reconstruction %d starting" % (islit), flush = True)

r2, r_space = phase.hio_er(
    intensity, support, loop_hio = 5000, loop_er = 500, 
    iteration = 1, show = 1, internal = 0
    )

print("finished", flush = True)

plt.imshow(np.abs(r_space))

import pickle

f = open("result_pc_slit%d0_set%d_test_1mode.pkl" % (islit, iset), 'wb')
pickle.dump([r2, r_space], f, True)
f.close()
