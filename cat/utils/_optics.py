#-----------------------------------------------------------------------------#
# Copyright (c) 2021 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS HXS (B4) xuhan@ihep.ac.cn"
__date__     = "Date : 04.01.2021"
__version__  = "beta-1.0"


"""
_optics: The base class of optics.

Functions: _locate - return the nearest location of value among ticks
           
Classes  : _optic_plane - the geometry structure of optics
"""

#-----------------------------------------------------------------------------#
# library

import numpy as np
import h5py  as h5

import pickle

from scipy import interpolate
from copy  import deepcopy

#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function

def _locate(ticks, value):
    
    """
    Return the nearest location of value among ticks
    
    Args: ticks - numpy array of data.
          value - a value to be located
    """
    
    if value > np.max(ticks) or value < np.min(ticks):
        raise ValueError("The given value is out of range.")
    else:
        return np.argmin(np.abs(ticks - value))

#-----------------------------------------------------------------------------#
# class

# Todo: delete this class.
class _optic_plane(object):
    
    """
    The basic geometry parameters of optic plane.
    """
    
    def __init__(self, xcoor, ycoor, location):
        
        # the cooridnate of wavefront
        
        self.xstart, self.xend, self.xcount = xcoor
        self.ystart, self.yend, self.ycount = ycoor
        
        self.xcount = int(self.xcount)
        self.ycount = int(self.ycount)
        
        self.xstart = self.xstart
        self.ystart = self.ystart
        
        self.xend = self.xend
        self.yend = self.yend
        
        self.xtick = np.linspace(self.xstart, self.xend, int(xcoor[2]))
        self.ytick = np.linspace(self.ystart, self.yend, int(ycoor[2]))
        
        self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
        
        self.xpixel = np.abs(xcoor[0] - xcoor[1])/xcoor[2]
        self.ypixel = np.abs(ycoor[0] - ycoor[1])/ycoor[2]
        
        self.mask = np.ones((self.ycount, self.xcount))
        self.plane = np.zeros((self.ycount, self.xcount), dtype = complex)
        
        self.location = location

class _optics(object):
    
    """
    The geometry parameters and modulation methods of optic plane.
    
    Noting: the unit of length is metre (m.)
    
    methods: remap       - Interpolate the coherent mode data and optic plane. 
             center      - put coherent mode to center.
             remap_plane - interpolate the optic plane (not coherent mode).
             mask        - construct mask to the optic plane and coherent mode.
             slit        - construct and apply a slit.
             expand      - zero padding the optic plane and coherent mode.
             hermite_n   - constructe 2d hermite mode.
             cal_csd     - calculate the CSD of the optics.
             cal_i       - calculate the intensity of the optic plane.
             error2d     - apply a 2d phase error.
             cmd_area    - calculate the CSD of the optics within the 
                           specified area.
             save        - save the optic into a .pkl file.
    """
    
    def __init__(self, optic = None, name = "optic", 
                 n_vector = 0, position = 0):
            
        self.xstart = np.copy(optic.xstart)
        self.xend   = np.copy(optic.xend)
        self.xcount = np.copy(optic.xcount)
        
        self.ystart = np.copy(optic.ystart)
        self.yend   = np.copy(optic.yend)
        self.ycount = np.copy(optic.ycount)
        
        self.xtick  = np.copy(optic.xtick)
        self.ytick  = np.copy(optic.ytick)
        
        self.gridx  = np.copy(optic.gridx)
        self.gridy  = np.copy(optic.gridy)
        
        self.xpixel = np.copy(optic.xpixel)
        self.ypixel = np.copy(optic.ypixel)
        self.mask   = np.ones((self.ycount, self.xcount))
        self.n      = np.copy(optic.n)
        
        self.position   = position
        self.name = name
        
        self.wavelength = np.copy(optic.wavelength)
        self.ratio      = np.copy(optic.ratio)
        self.cmode      = list()
        
        for i in range(self.n): 
            self.cmode.append(
                np.ones((self.ycount, self.xcount), dtype = complex)
                )
        
        self.ori_cmode  = list()
        
        self.csd2dx = np.empty(0)
        self.csd2dy = np.empty(0)
        self.csd1dx = np.empty(0)
        self.csd1dy = np.empty(0)
        
        # cal spectral degree of coherence
        
        self.miu2dx = np.empty(0)
        self.miu2dy = np.empty(0)
        self.miu1dx = np.empty(0)
        self.miu1dy = np.empty(0)
        
        self.clx = float()
        self.cly = float()
        
        self.intensity  = np.empty(0)
        self.fwhmx = float()
        self.fwhmy = float()
        
    def remap(self, xpixel, ypixel):
        
        """
        Interpolate the coherent mode data. To satisfiy the sampling limit.
        
        Args: xpixel  - the sampling density along x.
              ypixel  - the sampling density along y.
              
        Return: None.
        """
        
        self.xpixel = xpixel
        self.ypixel = ypixel
        
        xcount = int((self.xend - self.xstart)/xpixel)
        ycount = int((self.yend - self.ystart)/ypixel)
        
        # the size of the remapped data is odd. 
        # reason: some annoying shift of pixels after FFT (numpy and scipy) has
        #         been observed. Using scipy FFT and odd size plane can reduce
        #         this effect to half pxiel.
        #         Therefore, always remap before propagate with FFT.
        
        if xcount % 2 == 0:
            xcount = xcount + 1
        
        if ycount % 2 == 0:
            ycount = ycount + 1
        
        # remap the plane
            
        xtick = np.linspace(self.xstart, self.xend, xcount)
        ytick = np.linspace(self.ystart, self.yend, ycount)

        for i in range(self.n):
        
            freal = interpolate.interp2d(
                self.xtick, self.ytick, np.real(self.cmode[i]), 
                kind = 'cubic')
            fimag = interpolate.interp2d(
                self.xtick, self.ytick, np.imag(self.cmode[i]),
                kind = 'cubic')
    
            self.cmode[i] = freal(xtick, ytick) + 1j*fimag(xtick, ytick)
        
        self.xcount = xcount
        self.ycount = ycount
        self.xtick = xtick
        self.ytick = ytick
        self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
        self.mask = np.ones((self.ycount, self.xcount))
            
    def remap_plane(self, xpixel, ypixel):
        
        """
        Interpolate the optic plane (not coheret mode). This method is used to
        adjust the geometry requirement of receive optic plane.
        
        Args: xpixel  - the sampling density along x.
              ypixel  - the sampling density along y.
              
        Return: None.
        """
        
        self.xpixel = xpixel
        self.ypixel = ypixel
        
        xcount = int((self.xend - self.xstart)/xpixel)
        ycount = int((self.yend - self.ystart)/ypixel)
        
        # the size of the remapped data is odd. 
        # reason: some annoying shift of pixels after FFT (numpy and scipy) has
        #         been observed. Using scipy FFT and odd size plane can reduce
        #         this effect to half pxiel.
        #         Therefore, always remap before propagate with FFT.
        
        if xcount % 2 == 0:
            xcount = xcount + 1
        
        if ycount % 2 == 0:
            ycount = ycount + 1
            
        # remap the plane
            
        xtick = np.linspace(self.xstart, self.xend, xcount)
        ytick = np.linspace(self.ystart, self.yend, ycount)

        self.xcount = xcount
        self.ycount = ycount
        self.xtick = xtick
        self.ytick = ytick
        self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
        self.mask = np.ones((self.ycount, self.xcount))

        for i in range(self.n): 
            self.cmode.append(np.ones((self.ycount, self.xcount)))
            
    def add_mask(self, xcoor = None, ycoor = None, r = None, s = "b"):
         
        """
        Construct a mask. The shape could be box or circle.
        
        The data outside the mask will be set to zeros.
        
        Args: xcoor - [xstart, xend].
              ycoor - [ystart, yend].
              r     - radicus of mask.
              s     - shape: "b" for box; "c" for circle.
              
        Return: None.
        """
        
        self.mask = np.zeros((self.ycount, self.xcount))
        
        if s == "b":
        
            # find the location of mask area.
            
            locxs = _locate(self.xtick, xcoor[0])
            locys = _locate(self.ytick, ycoor[0])
            locxe = _locate(self.xtick, xcoor[1])
            locye = _locate(self.ytick, ycoor[1])
            
            # construct mask
            
            self.mask[locys : locye, locxs : locxe] = 1
        
        elif s == "c":
            
            rad = np.sqrt(self.gridx**2 + self.gridy**2)
            
            # distance < r is set as unmasked
            
            self.mask[rad < r] = 1
            
        for i in range(self.n):
            self.cmode[i] = self.cmode[i] * self.mask
    
    def slit(self, xcoor = None, ycoor = None):
        
        """
        Construct and apply a slit. The data outside the slit will be deleted.
        
        Args: xcoor - [xstart, xend].
              ycoor - [ystart, yend].
        """
        
        if xcoor is None:
            locxs = 0
            locxe = self.xcount
        else:
            locxs = _locate(self.xtick, xcoor[0])
            locxe = _locate(self.xtick, xcoor[1])          
            
        if ycoor is None:
            locys = 0
            locye = self.ycount
        else:
            locys = _locate(self.ytick, ycoor[0])
            locye = _locate(self.ytick, ycoor[1])
    
        self.xstart = xcoor[0]
        self.xend   = xcoor[1]
        self.ystart = ycoor[0]
        self.yend   = ycoor[1]
        
        self.xcount = int(locxe - locxs)
        self.ycount = int(locye - locys)
        
        self.xtick = np.linspace(self.xstart, self.xend, self.xcount)
        self.ytick = np.linspace(self.ystart, self.yend, self.ycount)
        
        self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
        self.mask = np.ones((self.ycount, self.xcount))
        
        for i in range(self.n):
            self.cmode[i] = self.cmode[i][locys : locye, locxs : locxe]
            
    def expand(self, xcoor = None, ycoor = None):
        
        """
        Zero padding the optic plane to satify the requirement of 
        Fresnel diffraction.
        
        Args: xcoor - [xstart, xend].
              ycoor - [ystart, yend].
              
        Return: None.
        """
        
        eplx = int(np.abs(xcoor[0] - self.xstart)/self.xpixel)
        eprx = int(np.abs(xcoor[1] - self.xend)/self.xpixel)
        eply = int(np.abs(ycoor[0] - self.ystart)/self.ypixel)
        epry = int(np.abs(ycoor[1] - self.yend)/self.ypixel)
        
        xcount = eplx + eprx + self.xcount
        ycount = eply + epry + self.ycount
        
        xstart = self.xstart - eplx * self.xpixel
        xend = eprx * self.xpixel + self.xend
        ystart = self.ystart - eply * self.ypixel
        yend = epry * self.ypixel + self.yend
        
        cmode = np.zeros((ycount, xcount), dtype = complex)
        
        for i in range(self.n):
            cmode[eply : eply + self.ycount, 
                  eplx : eplx + self.xcount] = self.cmode[i]
            self.cmode[i] = np.copy(cmode)
            cmode = np.zeros((ycount, xcount), dtype = complex)
        
        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend
        self.xcount = xcount
        self.ycount = ycount
        self.xtick = np.linspace(xstart, xend, xcount)
        self.ytick = np.linspace(ystart, yend, ycount)
        self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
        
        self.mask = np.ones((self.ycount, self.xcount))
            
    def cal_csd(self):
        
        """
        Calculate CSD of the optics. Calcuate the csd of the optic plane 
        before the plot (1d and 2d).
        
        Noting: Usually, the size of CSD is very large. Remap the data to a 
                smaller size.
        
        Args: None
        
        Return: None
        """
        
        # get center slice
        
        cmodex = np.zeros((self.ycount, self.xcount), dtype = np.complex128)
        cmodey = np.zeros((self.ycount, self.xcount), dtype = np.complex128)
        
        for i in range(self.n):
            
            ratio = np.sqrt(self.ratio[i])
            cmodex[i, :] = self.cmode[i][int(self.ycount/2), :] * ratio
            cmodey[i, :] = self.cmode[i][:, int(self.xcount/2)] * ratio
        
        # calcualte csd
        
        self.csd2dx = np.dot(cmodex.T.conj(), cmodex)
        self.csd2dy = np.dot(cmodey.T.conj(), cmodey)
        self.csd1dx = np.abs(np.diag(np.fliplr(self.csd2dx)))
        self.csd1dy = np.abs(np.diag(np.fliplr(self.csd2dy)))
        
        # cal spectral degree of coherence
        
        i0x = np.diag(np.abs(self.csd2dx))
        i0y = np.diag(np.abs(self.csd2dy))
        
        i0x2d = np.zeros((self.xcount, self.xcount))
        i0y2d = np.zeros((self.ycount, self.ycount))
        
        for i in range(self.xcount): i0x2d[i, :] = i0x
        for i in range(self.ycount): i0y2d[i, :] = i0y
        
        self.miu2dx = self.csd2dx / np.sqrt(i0x2d * i0x2d.T)
        self.miu2dy = self.csd2dy / np.sqrt(i0y2d * i0y2d.T)
        
        self.miu1dx = np.abs(np.diag(np.fliplr(self.miu2dx)))
        self.miu1dy = np.abs(np.diag(np.fliplr(self.miu2dy)))
        
        # calculate coherent length
        
        sort_x = np.argsort(np.abs(self.miu1dx - np.max(self.miu1dx)/2))
        sort_y = np.argsort(np.abs(self.miu1dy - np.max(self.miu1dy)/2))
        
        self.clx = np.abs(sort_x[0] - sort_x[1]) * self.xpixel/2
        self.cly = np.abs(sort_y[0] - sort_y[1]) * self.ypixel/2
        
    def cal_i(self):
        
        """
        Calculate the intensity of the optics. 
        
        Args: None
        
        Return: None
        """
        
        self.intensity = np.zeros((self.ycount, self.xcount))
        
        for i, ic in enumerate(self.cmode):
            
            self.intensity = self.intensity + self.ratio[i]*np.abs(ic)**2
            
        ix = np.sum(self.intensity, 0)
        iy = np.sum(self.intensity, 1)
        
        sort_x = np.argsort(np.abs(ix - np.max(ix)/2))
        sort_y = np.argsort(np.abs(iy - np.max(iy)/2))
        
        self.fwhmx = np.abs(sort_x[0] - sort_x[1]) * self.xpixel
        self.fwhmy = np.abs(sort_y[0] - sort_y[1]) * self.ypixel
    
    def error2d(self, error):
        
        """
        Add phase error to this optic plane.
        
        Args: er - 2d error phase data
              
        Return: None.
        """
        
        self.er_phase = error
        
        self.cmode = [np.abs(self.cmode[i]) * 
                      np.exp(1j*(self.er_phase + np.angle(self.cmode[i]))) 
                      for i in range(self.n)] 
            
    def cmd(self, xcoor = None, ycoor = None):
        
        """
        Re-perform coherent mode decomposition. 
        
        Noting: memory and time costing! A better method is under developping.
        
        Args: xcoor - [xstart, xend].
              ycoor - [ystart, yend].
              
        Return: None.
        """
        
        if xcoor is None:
            locxs, locxe = 0, self.xcount
        else:
            locxs = _locate(self.xtick, xcoor[0])
            locxe = _locate(self.xtick, xcoor[1])
        
        if ycoor is None:
            locys, locye = 0, self.ycount
        else:
            locys = _locate(self.ytick, ycoor[0])
            locye = _locate(self.ytick, ycoor[1])
        
        re_cmode = np.array(self.cmode)[:, locys : locye, locxs : locxe]
        xcount, ycount = re_cmode.shape[2], re_cmode.shape[1]
        cmode = np.reshape(re_cmode, (self.n, xcount * ycount))
        
        for i in range(self.n):
            cmode[i] = cmode[i] * np.sqrt(self.ratio[i])
        
        csd = np.dot(cmode.T.conj(), cmode)
        
        del cmode
        
        from scipy.sparse import linalg
        
        eig_value, eig_vector = linalg.eigsh(csd, k = 2*self.n + 1)
        eig_vector = np.reshape(eig_vector, (ycount, xcount, 2*self.n + 1))
        cmode = np.zeros((self.ycount, self.xcount), 
                         dtype = np.complex128)
        
        del csd
        
        for i in range(self.n):
            
            self.cmode[i] = np.copy(cmode)
            self.cmode[i][locys : locye, locxs : locxe] = eig_vector[:, :, i]
            self.ratio[i] = eig_value[i]
    
    def save(self):
        
        """
        Save all the properites to a pickle.
        
        Noting: please reset the name of the optic plane before saving.
        
        Args: None.
        
        Return: None.
        """
        
        pickle.dump(self, open(self.name + '.pkl', 'wb'), True)
            
