#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_hermite_gaussian: the construction of hermite gaussian modes and csd.

functions: _gaussian_csd1d - construct 1d hermite gaussian csd
           
classes  : _hermite_gaussian    - hermite_gaussian mode
           _gaussian_schell_csd - get csd from gaussian schell
"""


#-----------------------------------------------------------------------------#
# library

import numpy as np

from scipy.linalg import eig    as acc_eig
from scipy.sparse import linalg as aro_linalg

#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function

def _gaussian_csd1d(A, sigma_miu, sigma_s, x):
    
    """
    ---------------------------------------------------------------------------
    construct 1d hermite gaussian csd.
    
    args: A         - amplitude.
          sigma_miu - the gaussian sigma of coherence.
          sigma_s   - the gaussian sigma of spectral intensity.
          x         - 1d range.
    
    return: 1d csd of gaussain schell.   
    --------------------------------------------------------------------------- 
    """
    
    x1, x2 = np.meshgrid(x, x)
    spectral_density1 = A * np.exp(-x1**2/(2*sigma_s**2))
    spectral_density2 = A * np.exp(-x2**2/(2*sigma_s**2))
    miu = np.exp(-(x1 - x2)**2/(2*sigma_miu**2))
    
    return (np.sqrt(spectral_density1) * 
            np.sqrt(spectral_density2) * 
            miu)

#-----------------------------------------------------------------------------#
# class
    
class _hermite_gaussian(object):
    
    """
    ---------------------------------------------------------------------------
    construct 1d and 2d hermite gaussian modes.
    
    methods: ratio     - ratio of modes (normalized by maximum).
             hermite   - constructe hermite mode with hermite function.
             hermite_x - constructe hermite mode in axis x.
             hermite_y - constructe hermite mode in axis y.
             hermite_n - constructe 2d hermite mode.
    ---------------------------------------------------------------------------
    """
    
    def __init__(self, 
                 amplitude = 1, 
                 sigma_sx =3e-6, sigma_sy = 3e-6,
                 sigma_miu_x = 1e-6, sigma_miu_y = 3e-6,
                 xleft = -2e-5, xright = 2e-5,
                 yleft = -2e-5, yright = 2e-5,
                 nx = 128, ny = 128):
        
        self.amplitude = amplitude
        
        self.ax = 1/(4*sigma_sx**2)
        self.bx = 1/(2*sigma_miu_x**2)
        self.cx = np.sqrt(self.ax**2 + 2*self.ax*self.bx)
        
        self.ay = 1/(4*sigma_sy**2)
        self.by = 1/(2*sigma_miu_y**2)
        self.cy = np.sqrt(self.ay**2 + 2*self.ay*self.by)
        
        self.x = np.linspace(xleft, xright, nx)
        self.y = np.linspace(yleft, yright, ny)
        self.nx = int(nx)
        self.ny = int(ny)
        
    def ratio(self, n, nx, a, b, c, amplitude):
        
        """
        ---------------------------------------------------------------------------
        ratio of modes (normalized by maximum). 
        ref: "coherent-mode representations in optics"
        
        args: n         - 1d hermite mode index.
              nx        - sampling.
              a         - ref.equation 1.61
              b         - ref.equation 1.61
              c         - ref.equation 1.61
              amplitude - the gaussian sigma of spectral intensity.
        
        return: ratio of mode  
        ---------------------------------------------------------------------------
        """
        
        # ref.equation 1.60
        
        ratio = [((amplitude * np.sqrt(np.pi/(a + b + c)) * 
                   (b/(a + b + c))**i)) 
                 for i in range(nx)]
        ratio = np.array(ratio) / np.max(ratio)
        
        return ratio[n]
        
    def hermite(self, n, nx, a, b, c, amplitude, x):
        
        """
        ---------------------------------------------------------------------------
        constructe 1d hermite mode with hermite function.
        ref: "coherent-mode representations in optics"
        
        args: n         - 1d hermite mode index.
              nx        - sampling.
              a         - ref.equation 1.61
              b         - ref.equation 1.61
              c         - ref.equation 1.61
              amplitude - the gaussian sigma of spectral intensity.
        
        return: 1d hermite mode
        ---------------------------------------------------------------------------
        """
        
        from scipy import special
        
        ratio = self.ratio(n, nx, a, b, c, amplitude)
        
        # ref.equation 1.59
        
        hermite = ((2*c / np.pi)**0.25 * 
                   (1 / np.sqrt(2**n * special.factorial(n))) *
                   special.hermite(n)(x * np.sqrt(2*c)) *
                   np.exp(-c * x**2)
                   )
        
        return ratio, hermite
    
    def hermite_x(self, n):
        
        """
        ---------------------------------------------------------------------------
        constructe hermite mode in axis x.
        
        args: n - 1d hermite mode index.

        return: x hermite mode
        ---------------------------------------------------------------------------
        """
        
        return self.hermite(n, self.nx, self.ax, self.bx, self.cx, 
                            self.amplitude, self.x)
    
    def hermite_y(self, n):
        
        """
        ---------------------------------------------------------------------------
        constructe hermite mode in axis y.
        
        args: n - 1d hermite mode index.

        return: y hermite mode
        ---------------------------------------------------------------------------
        """
        
        return self.hermite(n, self.ny, self.ay, self.by, self.cy, 
                            self.amplitude, self.y)
    
    def hermite_n(self, n0, n1):
        
        """
        ---------------------------------------------------------------------------
        constructe 2d hermite mode.
        
        args: n0 - axis x hermite mode index.
              n1 - axis y hermite mode index.
              
        return: 2d hermite mode
        ---------------------------------------------------------------------------
        """
        
        hermite_0 = self.hermite_x(n0)
        hermite_1 = self.hermite_y(n1)
        
        ratio = hermite_0[0] * hermite_1[0]
        hermite = np.dot(np.matrix(hermite_1[1]).T, np.matrix(hermite_0[1]))
        res = [ratio, hermite/np.max(hermite)]
        
        return res

#-------------------------------------------------------------
        
class _gaussian_schell_csd(object):
    
    """
    ---------------------------------------------------------------------------
    get csd from gaussian schell.
    
    methods: csdx  - construct gaussain schell csd in axis x.
             csdy  - constructe gaussian schell csd in axis y.
             csd   - 2d gaussian schell csd.
             modex - matrix decompostion of csdx.
             modey - matrix decompostion of csdy.
             mode  - matrix decompostion of csd.
    ---------------------------------------------------------------------------
    """
    
    def __init__(self, 
                 amplitude = 1, 
                 sigma_sx =3e-6, sigma_sy = 3e-6,
                 sigma_miu_x = 1e-6, sigma_miu_y = 3e-6,
                 xleft = -2e-5, xright = 2e-5,
                 yleft = -2e-5, yright = 2e-5,
                 nx = 128, ny = 128):
        
        self.a = amplitude
        self.s_miux = sigma_miu_x
        self.s_miuy = sigma_miu_y
        self.sx = sigma_sx
        self.sy = sigma_sy
        self.x = np.linspace(xleft, xright, nx)
        self.y = np.linspace(yleft, yright, ny)
        self.mode = list()
        self.ratio = list()
        self.nx = int(nx)
        self.ny = int(ny)
        
    def csdx(self):
        
        """
        ---------------------------------------------------------------------------
        construct gaussain schell csd in axis x.
        
        args: none.
        
        return: csdx 
        ---------------------------------------------------------------------------
        """
        
        return _gaussian_csd1d(self.a, self.s_miux, self.sx, self.x)
    
    def csdy(self):
        
        """
        ---------------------------------------------------------------------------
        construct gaussain schell csd in axis y.
        
        args: None.
        
        return: csdy.
        ---------------------------------------------------------------------------
        """
        
        return _gaussian_csd1d(self.a, self.s_miuy, self.sy, self.y)
    
    def csd(self):
        
        """
        ---------------------------------------------------------------------------
        construct gaussain schell 2d csd from csdx and csdy.
        
        args: None.
        
        return: csd2d 
        ---------------------------------------------------------------------------
        """
        
        csdx = self.csdx()
        csdy = self.csdy()
        csd2d = np.zeros((self.nx * self.ny, self.nx * self.ny), 
                         dtype = np.complex)
        for i0 in range(self.nx):
            for i1 in range(self.ny):
                csd2d[i0 * self.nx : (i0 + 1) * self.nx,
                      i1 * self.nx : (i1 + 1) * self.nx] = csdy[i0, i1] * csdx
        return csd2d
    
    def modex(self, n):
        
        """
        ---------------------------------------------------------------------------
        matrix decompostion of csdx, analysis matrix decompostion.
        
        args: n - hermite mode index.
        
        return: res - [ratio of the mode, mode]
        ---------------------------------------------------------------------------
        """
        
        ratio, x_mode = acc_eig(self.csdx())
        ratio = ratio / np.max(ratio)
        res = [ratio[int(n)], x_mode[:, int(n)]]
        
        return res
    
    def modey(self, n):
        
        """
        ---------------------------------------------------------------------------
        matrix decompostion of csdy, analysis matrix decompostion.
        
        args: n - hermite mode index.
        
        return: res - [ratio of the mode, mode]
        ---------------------------------------------------------------------------
        """
        
        ratio, y_mode = acc_eig(self.csdy())
        ratio = ratio / np.max(ratio)
        res = ratio[int(n)], y_mode[:, int(n)]
        
        return res
    
    def cal_mode(self, n, n_vector = 200):
        
        """
        ---------------------------------------------------------------------------
        matrix decompostion of csd2d, analysis. aronoldi method is used.
        
        args: n        - hermite mode index.
              n_vector - approxmite step.
        
        return: res - [ratio of the mode, mode (normalised to 1)].
        ---------------------------------------------------------------------------
        """
        
        if len(self.mode) == 0:

            self.ratio, mode = aro_linalg.eigsh(self.csd(), k = int(n_vector))
            self.ratio = self.ratio / np.sum(self.ratio)
            self.mode = [
                np.reshape(mode[:, int(i)], (self.nx, self.ny)) 
                for i in range(n_vector)
                ]
        
        return [self.ratio[n], self.mode[n]]

#-----------------------------------------------------------------------------#

    