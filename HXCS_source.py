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

_ncount = 6

#------------------------------------------------------------------------------
# source

sr0 = source2(
    file_name = 'b4_srw2_12400.h5', name = 'source', n_vector = _ncount
    )

# plot coherent modes of source
tool.plot_optic(sr0, t = 'mode', n = (2, 3))

