#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"

#-----------

import numpy as np
import matplotlib.pyplot as plt
import h5py as h

from cat.optics import source2, screen, ideal_lens
from cat.propagate import propagate_s
from cat import tool

#----------------------
# plot modes

esrf_ebs = source2(file_name = "ebs_cat.h5", name = "ebs", n_vector = 4)
esrf_ebs.remap(0.5e-6, 0.5e-6)

crl = ideal_lens(optics = esrf_ebs, n = 4, location = 10,
                 xfocus = 5,  yfocus = 5)
propagate_s(esrf_ebs, crl)

sr = screen(optics = esrf_ebs, n = 4, location = 20)
propagate_s(crl, sr)


sr.xtick = sr.xtick - 4.37e-6
sr.slit(xcoor = [-3e-5, 3e-5], ycoor = [-3e-5, 3e-5], t = 0)
tool.plot_optic(sr, t = 'mode', n = (2, 2))

# plot ratio

with h.File("ebs_cat.h5", 'a') as f:

    value = np.array(f["coherence/eig_value"])
    
    fig, ax = plt.subplots(1, 1, figsize = (12, 6))
    ax.scatter(range(500), value[0 : 500]**2 / np.sum(value[0 : 500]**2))
    ax.set_title("%.3f percent" % (100 * value[0]**2 / np.sum(value[0 : 500]**2)))