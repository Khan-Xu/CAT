#-----------------------------------------------------------------------------#
# Copyright (c) 2021 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - HEPS HXS (B4) xuhan@ihep.ac.cn"
__date__     = "Date : 04.01.2021"
__version__  = "beta-1.0"


"""
tool: support and visualization functions.

Functions: _fresnel    - DFFT fresenl propagation.
           propagate_s - single process coherent mode propagation
           propagate_m - multi-process coehrent mode propagation

Classes  : None
"""

#-----------------------------------------------------------------------------#
# library

import numpy             as np
import h5py              as h5
import matplotlib.pyplot as plt

from copy  import deepcopy
from scipy import interpolate

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
    
def _rotate(matrix, u = 0):
    
    """
    Rotate a 2d matrix
    
    Args: matrix - 2d matrix to rotate
          u      - rotation angle
    """
    
    xcount, ycount = np.shape(np.matrix(matrix))
    xtick = np.arange(xcount) - xcount/2
    ytick = np.arange(ycount) - ycount/2
    gridx, gridy = np.meshgrid(xtick, ytick)
    
    xcoor = np.reshape(gridx, (xcount*ycount, 1))[:, 0]
    ycoor = np.reshape(gridy, (xcount*ycount, 1))[:, 0]
    data = np.reshape(matrix, (xcount*ycount, 1))[:, 0]
    
    coor = np.zeros((2, xcount*ycount))
    coor[0, :] = xcoor
    coor[1, :] = ycoor
    eular = np.matrix([[np.cos(u), -np.sin(u)], [np.sin(u), np.cos(u)]])
    coor = np.dot(eular, np.matrix(coor))
    rcoor = np.zeros((xcount*ycount, 2))
    rcoor[:, 0] = coor[0, :]
    rcoor[:, 1] = coor[1, :]
    
    greal = interpolate.griddata(rcoor, 
                                 np.real(data), (gridx, gridy), 
                                 method = 'cubic')
    gimag = interpolate.griddata(rcoor, 
                                 np.imag(data), (gridx, gridy), 
                                 method = 'cubic')

    rotated_data = greal + 1j*gimag
    
    return rotated_data

def _gaussfit(x, y, mean, sigma):
    
    """
    Gauss fitting by scipy curve_fit
    
    Args: x     - the x-coordinate of data.
          y     - the y-coordinate of data.
          mean  - the guessed mean of data.
          sigma - the gaussed sigma of data.
          
    Return: the fitted y data and parameters.
    """
    
    def gauss(x, a, x0, sigma):
        return a*np.exp(-(x - x0)**2/(2*sigma**2))
    
    from scipy.optimize import curve_fit
    
    popt, pcov = curve_fit(gauss, x, y, p0 = [1, mean, sigma])
    
    fity = gauss(x, *popt)
    
    return fity, popt

#-----------------------------------------------------------------------------#

def read_srw(file_name, h_range = None, v_range = None, 
             mode = "intensity", save = 0):
    
    """
    Read and plot srw data (intensity and moi data).
    
    Args: file_name - construct a suqare mask.
          h_range   - [h_start, h_end] of specified range along h axis.
          v_range   - [v_start, v_end] of specified range along v axis.
          mode      - intensity, moix and moiy.
    
    Return: h, v range and data (abs)
    """
    
    def find_divide(string):
        
        """
        Divide string with #.
        
        Args: string
        
        Return: divided string
        """
    
        return string[string.find('#', 0) + 1 : string.find('#', 1)]
    
    def find_near(data, value):
        
        """
        find the location of nearest value.
        
        Args: data  - the data range to locate.
              value - the value to be located.
        
        Return: location of the value.
        """
        
        idx = (np.abs(data - value)).argmin()
        
        return idx
    
    with open(file_name, 'r') as file:
        
        # read data head
        
        all_data = [line.strip() for line in file.readlines()]
        
        # plot intensity
        
        if mode == "intensity" or mode == "i1d":
            
            # read lable. energy range, h and v range.
            
            label        = all_data[0][1:28]
            energy_start = float(find_divide(all_data[1]))
            energy_end   = float(find_divide(all_data[2]))
            
            h_start = float(find_divide(all_data[4]))
            h_end   = float(find_divide(all_data[5]))
            h_count = int(find_divide(all_data[6]))
            
            v_start = float(find_divide(all_data[7]))
            v_end   = float(find_divide(all_data[8]))
            v_count = int(find_divide(all_data[9]))
            
            # read all the data and reshape to matrix.
            
            data = np.array(list(map(float, all_data[11:])))
            data = np.reshape(data, [h_count, v_count])
        
        # plot moix
        
        elif mode == "moix":
            
            # read lable. energy range, h and v range.
            
            label        = all_data[0][1:26]
            energy_start = float(find_divide(all_data[1]))
            energy_end   = float(find_divide(all_data[2]))
            
            h_start = float(find_divide(all_data[4]))
            h_end   = float(find_divide(all_data[5]))
            h_count = int(find_divide(all_data[6]))
            
            v_start = deepcopy(h_start)
            v_end   = deepcopy(h_end)
            v_count = deepcopy(h_count)
            
            # read all the data and reshape to 3d matrix (real and imag)
            
            data = np.array(list(map(float, all_data[11:])))
            data = np.reshape(data, [h_count, v_count, 2])
            data = np.abs(data[:, :, 0] + 1j*data[:, :, 1])
        
        # plot moiy
        
        elif mode == "moiy":
            
            # read label. energy range, h and v range.
            
            label        = all_data[0][1:26]
            energy_start = float(find_divide(all_data[1]))
            energy_end   = float(find_divide(all_data[2]))
            
            v_start = float(find_divide(all_data[4]))
            v_end   = float(find_divide(all_data[5]))
            v_count = int(find_divide(all_data[6]))
            
            h_start = deepcopy(v_start)
            h_end   = deepcopy(v_end)
            h_count = deepcopy(v_count)
            
            # read all the data and reshape to 3d matrix (real and imag)
            
            data = np.array(list(map(float, all_data[11:])))
            data = np.reshape(data, [h_count, v_count, 2])
            data = np.abs(data[:, :, 0] + 1j*data[:, :, 1])
            
    h = np.linspace(h_start, h_end, h_count)*1e6
    v = np.linspace(v_start, v_end, v_count)*1e6
    
    # if new data range is speicfied. Re-arange the data.
    
    if h_range is not None:
        
        idx_hl = find_near(h, h_range[0])
        idx_hr = find_near(h, h_range[1])
        
    else:
        
        idx_hl = 0
        idx_hr = np.shape(h)[0]
    
    if v_range is not None:
        
        idx_vl = find_near(v, v_range[0])
        idx_vr = find_near(v, v_range[1])
    
    else:
        
        idx_vl = 0
        idx_vr = np.shape(v)[0]
    
    h_re = h[idx_hl : idx_hr]
    v_re = v[idx_vl : idx_vr]
    data = data[idx_hl : idx_hr, idx_vl : idx_vr]
    
    data_x = np.sum(data, 0)/np.max(np.sum(data, 0))
    data_y = np.sum(data, 1)/np.max(np.sum(data, 1))
    
    sort_x = np.argsort(np.abs(data_x - 0.5))
    sort_y = np.argsort(np.abs(data_y - 0.5))
    
    clx = np.abs(h_re[sort_x[0]]) + np.abs(h_re[sort_y[1]])
    cly = np.abs(v_re[sort_x[0]]) + np.abs(v_re[sort_y[1]])
    
    # plot intenisty, sumx and sumy
    
    if mode == "intensity":
        
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        
        c = ax0.pcolor(h_re, v_re, data)
        ax0.set_title(label)
        fig.colorbar(c, ax = ax0)
        
        ax1.plot(h_re, data_x)
        ax1.set_title("intenisty_x FWHM %.2f um" % (clx))

        ax2.plot(v_re, data_y)
        ax2.set_title("intensity_y FWHM %.2f um" % (cly))
        
        if save:
            
            if h_re.shape[0] > 300:
                
                func = interpolate.interp2d(h_re, v_re, data, kind = 'cubic')
                rx = np.linspace(h_re[0], h_re[-1], 300)
                ry = np.linspace(v_re[0], v_re[-1], 300)
                rd = func(rx, ry)
                
            else:
                
                rx = h_re
                ry = v_re
                rd = data
                
            x, y = np.meshgrid(rx, ry)              

            rdata = np.zeros((rx.shape[0]*ry.shape[0], 3))
            
            rdata[:, 0] = np.reshape(x, (rx.shape[0]*ry.shape[0], 1))[:, 0]
            rdata[:, 1] = np.reshape(y, (rx.shape[0]*ry.shape[0], 1))[:, 0]
            rdata[:, 2] = np.reshape(rd, (rx.shape[0]*ry.shape[0], 1))[:, 0]
        
            np.savetxt("srw_intensity.dat", rdata)
                
        return h_re, v_re, data, data_x, data_y
    
    elif mode == "i1d":
        
        fig, (ax0, ax1) = plt.subplots(1, 2)
        
        ax0.plot(h_re, data_x)
        ax0.set_title("intenisty_x FWHM %.2f um" % (clx))

        ax1.plot(v_re, data_y)
        ax1.set_title("intensity_y FWHM %.2f um" % (cly))
    
        if save:
            
            xdata = np.zeros((h_re.shape[0], 2))
            ydata = np.zeros((h_re.shape[0], 2))
            
            xdata[:, 0] = h_re
            xdata[:, 1] = data_x
            
            ydata[:, 0] = v_re
            ydata[:, 1] = data_y
            
            np.savetxt("i1dx.dat", xdata)
            np.savetxt("i1dy.dat", ydata)
                
        return h_re, v_re, data_x, data_y
    
    # plot moi and diag
    
    elif mode == "moix" or mode == "moiy":
        
        fig, (ax0, ax1) = plt.subplots(1, 2)
        
        c = ax0.pcolor(h_re, v_re, data)
        ax0.set_title(label)
        fig.colorbar(c, ax = ax0)

        
        moi1d = np.abs(np.diag(np.fliplr(data)))
        moi1d = moi1d/np.max(moi1d)
        ax1.plot(h_re, moi1d)
        ax1.set_title("moi_1d")
            
        if save:
                
            if h_re.shape[0] > 300:
                
                func = interpolate.interp2d(h_re, v_re, data, kind = 'cubic')
                rx = np.linspace(h_re[0], h_re[-1], 300)
                ry = np.linspace(v_re[0], v_re[-1], 300)
                rd = func(rx, ry)
                r1d = np.abs(np.diag(np.fliplr(rd)))
                r1d = r1d/np.max(r1d)
                
            else:
                
                rx = h_re
                ry = v_re
                rd = data
                r1d = r1d/np.max(r1d)
                
            x, y = np.meshgrid(rx, ry)              

            rdata = np.zeros((rx.shape[0]*ry.shape[0], 3))
            
            rdata[:, 0] = np.reshape(x, (rx.shape[0]*ry.shape[0], 1))[:, 0]
            rdata[:, 1] = np.reshape(y, (rx.shape[0]*ry.shape[0], 1))[:, 0]
            rdata[:, 2] = np.reshape(rd, (rx.shape[0]*ry.shape[0], 1))[:, 0]
                
            ldata = np.zeros((rx.shape[0], 2))
            ldata[:, 0] = rx
            ldata[:, 1] = r1d
                
            np.savetxt("srw_moi.dat", rdata)
            np.savetxt("srw_moi1d.dat", ldata)
                
        return h_re, v_re, data, moi1d

def plot_optic(optic, t = "intensity", n = (3, 3), save = 0):
    
    """
    Read and plot optic (intensity, coherent mode and moi data).
    
    Args: optic - optic or source class.
          t     - properties (intenisty, mode, csd) to plot.
          n     - if mode is chose to plot, n set the mode number (nx, ny).
          save  - save the image or not.
    Return: None.
    """
    
    # plot intenisty, sumx and sumy
    
    if t == "intensity":
        
        intensity = np.zeros((optic.ycount, optic.xcount))
        
        # calculate intensity
        
        for i, ic in enumerate(optic.cmode):
            intensity = intensity + optic.ratio[i] * np.abs(ic)**2
        
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (20, 5))
        c = ax0.pcolor(optic.xtick*1e6, optic.ytick*1e6, intensity)
        ax0.set_title("intensity")
        fig.colorbar(c, ax = ax0)
        
        ax1.plot(optic.xtick*1e6, 
                 np.sum(intensity, 0)/np.max(np.sum(intensity, 0)))
        ax1.set_title("intensity_x")

        ax2.plot(optic.ytick*1e6, 
                 np.sum(intensity, 1)/np.max(np.sum(intensity, 1)))
        ax2.set_title("intensity_y")
    
    # plot sumx and sumy of intensity
    
    if t == "i1d":
        
        intensity = np.zeros((optic.ycount, optic.xcount))
        
        # calculate intensity
        
        for i, ic in enumerate(optic.cmode):
            intensity = intensity + optic.ratio[i] * np.abs(ic)**2
        
        plt.figure(figsize = (5, 5))
    
        ix, = plt.plot(optic.xtick*1e6, 
                       np.sum(intensity, 0)/np.max(np.sum(intensity, 0)),
                       label = "intensity_x")
        iy, = plt.plot(optic.ytick*1e6, 
                       np.sum(intensity, 1)/np.max(np.sum(intensity, 1)),
                       label = "intensity_y")
        
        plt.legend(handles = [ix, iy])
    
    # plot coherent mode
    
    if t == "i1dgauss":
        
        intensity = np.zeros((optic.ycount, optic.xcount))
        
        # calculate intensity
        
        for i, ic in enumerate(optic.cmode):
            intensity = intensity + optic.ratio[i] * np.abs(ic)**2
          
        ix = np.sum(intensity, 0) / np.max(np.sum(intensity, 0))
        iy = np.sum(intensity, 1) / np.max(np.sum(intensity, 1))
        
        xx = optic.xtick * 1e6
        yy = optic.ytick * 1e6
        
        # calculate primary parameteres
        
        xmean = ix[_locate(ix, np.max(ix))]
        ymean = iy[_locate(iy, np.max(iy))]
        
        sort_x = np.argsort(np.abs(ix - np.max(ix)/2))
        sort_y = np.argsort(np.abs(iy - np.max(iy)/2))
        
        xsigma = np.abs(sort_x[0] - sort_x[1]) * optic.xpixel * 1e6/2.35
        ysigma = np.abs(sort_y[0] - sort_y[1]) * optic.ypixel * 1e6/2.35
        
        fitx, xpar = _gaussfit(xx, ix, xmean, xsigma)
        fity, ypar = _gaussfit(yy, iy, ymean, ysigma)
        
        print(xmean)
        print(xsigma)
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (15, 5))
    
        ax0.scatter(xx, ix)
        ax0.plot(xx, fitx)
        ax0.set_title("hor_sigma = %.3f um" % (xpar[2]))
        
        ax1.scatter(yy, iy)
        ax1.plot(yy, fity)
        ax1.set_title("ver_sigma = %.3f um" % (ypar[2]))
    
    if t == "mode":
        
        fig, axes = plt.subplots(int(n[0]), int(n[1]), 
                                 figsize = (4*int(n[0]), 4*int(n[1])))
        
        idx = 0
        
        # normalise the ratio
        
        ratio = optic.ratio/np.sum(optic.ratio)
        
        for i0 in range(int(n[0])):
            for i1 in range(int(n[1])):
                
                if int(n[0]) == 1:
                    axesi = axes[i1]
                else:
                    axesi = axes[i0, i1]
                
                axesi.pcolor(optic.xtick*1e6, optic.ytick*1e6, 
                             np.abs(optic.cmode[idx])**2)
                axesi.set_title("m=%d r=%.3f" % (idx, ratio[idx]))
                idx = idx + 1
    
    # plot csd
    
    if t == "csd":
        
        fig, axes = plt.subplots(2, 2, figsize = (12, 10))
        
        c = axes[0, 0].pcolor(optic.xtick*1e6, 
                              optic.ytick*1e6, np.abs(optic.csd2dx))
        axes[0, 0].set_title("MOI_x")
        fig.colorbar(c, ax = axes[0, 0])
        
        axes[0, 1].plot(
            2*optic.xtick[int(optic.xcount/2) : -1]*1e6, 
            optic.csd1dx[int(optic.xcount/2) : -1]/np.max(optic.csd1dx)
            )
        axes[0, 1].set_title("MOI_1dx")
        
        c = axes[1, 0].pcolor(optic.xtick*1e6, 
                              optic.ytick*1e6, np.abs(optic.csd2dy))
        axes[1, 0].set_title("MOI_y")
        fig.colorbar(c, ax = axes[1, 0])
        
        axes[1, 1].plot(
            2*optic.ytick[int(optic.ycount/2) : -1]*1e6, 
            optic.csd1dy[int(optic.ycount/2) : -1]/np.max(optic.csd1dy)
            )
        axes[1, 1].set_title("MOI_1dy")
        
    if t == "csd1d":
        
        plt.figure(figsize = (5, 5))
        
        moix, = plt.plot(optic.xtick*1e6, 
                         optic.csd1dx/np.max(optic.csd1dx), 
                         label = "MOI_1dx")
        moiy, = plt.plot(optic.ytick*1e6, 
                         optic.csd1dy/np.max(optic.csd1dy), 
                         label = "MOI_1dy")
        plt.legend(handles = [moix, moiy])
    
    if t == 'sdc':
        
        fig, axes = plt.subplots(2, 2, figsize = (10, 10))
        
        c = axes[0, 0].pcolor(optic.xtick*1e6, 
                              optic.ytick*1e6, np.abs(optic.miu2dx))
        axes[0, 0].set_title("SDC_x")
        fig.colorbar(c, ax = axes[0, 0])
        
        axes[0, 1].plot(2*optic.xtick[int(optic.xcount/2) : -1]*1e6, 
                        optic.miu1dx[int(optic.xcount/2) : -1])
        axes[0, 1].set_title("SDC_1dx")
        
        c = axes[1, 0].pcolor(optic.xtick*1e6, 
                              optic.ytick*1e6, np.abs(optic.miu2dy))
        axes[1, 0].set_title("SDC_y")
        fig.colorbar(c, ax = axes[1, 0])
        
        axes[1, 1].plot(2*optic.ytick[int(optic.ycount/2) : -1]*1e6, 
                        optic.miu1dy[int(optic.ycount/2) : -1])
        axes[1, 1].set_title("SDC_1dy")
        
    if t == 'ratio':
        
        plt.figure(figsize = (5, 5))
        plt.scatter(range(len(optic.ratio)), optic.ratio/np.sum(optic.ratio))
    
    
    plt.savefig(optic.name + t + ".png", dpi = 500)
        
def load_optic(optic_name):
    
    """
    Load saved optic class.
    
    Args: optic_name - the name of the optic to load.
    
    Return: the loaded optic class.
    """
    
    import pickle
    
    return pickle.load(open(optic_name + '.pkl', 'rb'))

def save_optic(optic):
    
    """
    Save optic class.
    
    Args: optic - optic to save.
    
    Return: None.
    """
    
    with h5.File(optic.name + '.h5', 'a') as f:
        
        optic_plane = f.create_group("optic_plane")
        
        optic_plane.create_dataset("name", data = optic.name)
        optic_plane.create_dataset("xstart", data = optic.xstart)
        optic_plane.create_dataset("xend", data = optic.xend)
        optic_plane.create_dataset("nx", data = optic.xcount)
        optic_plane.create_dataset("ystart", data = optic.ystart)
        optic_plane.create_dataset("yend", data = optic.yend)
        optic_plane.create_dataset("ny", data = optic.ycount)
        optic_plane.create_dataset("rx", data = optic.xtick)
        optic_plane.create_dataset("ry", data = optic.ytick)
        optic_plane.create_dataset("mesh_x", data = optic.mesh_x)
        optic_plane.create_dataset("mesh_y", data = optic.mesh_y)
        optic_plane.create_dataset("pixel_x", data = optic.pixel_x)
        optic_plane.create_dataset("pixel_y", data = optic.pixel_y)
        optic_plane.create_dataset("mask", data = optic.mask)
        optic_plane.create_dataset("position", data = optic.position)
        
        coherence = f.create_group("coherence")
        
        coherence.create_dataset("wavelength", data = optic.wavelength)
        coherence.create_dataset("ratio", data = optic.ratio)
        coherence.create_dataset("cmode", data = optic.cmode)
        coherence.create_dataset("csd2dx", data = optic.csd2dx)
        coherence.create_dataset("csd2dy", data = optic.csd2dy)
        coherence.create_dataset("csd1dx", data = optic.csd1dx)
        coherence.create_dataset("csd1dy", data = optic.csd1dy)
        coherence.create_dataset("phase_lens", data = optic.phase_lens)
        
def export_data(optic, t = 'intensity', c = 9):
    
    """
    Export data (intensity, coherent modes, csd, ratio) to three column data.
    
    Args: optic - optic to save.
          t     - properties (intenisty, mode, csd) to export.
          c     - the number of coherent modes to export.
    
    Return: None.
    """
    
    x = optic.xtick
    y = optic.ytick
    n = optic.xcount * optic.ycount
    
    x, y = np.meshgrid(x, y)
    x = np.reshape(x, (n, 1))[:, 0]
    y = np.reshape(y, (n, 1))[:, 0]
    
    if t == 'intensity':
        
        name = optic.name + '_' + t + '.dat'
        
        intensity = np.zeros((optic.xcount, optic.ycount), dtype = 'float')
        for i, ic in enumerate(optic.cmode):
            intensity = intensity + optic.ratio[i] * np.abs(ic)**2
        
        inten = np.zeros((n, 3), dtype = float)
        inten[:, 0] = x
        inten[:, 1] = y
        inten[:, 2] = np.reshape(intensity, (n, 1))[:, 0]
        
        np.savetxt(name, inten)
    
    if t == 'i1d':
        
        name = optic.name + '_' + t + '.dat'
        
        intensity = np.zeros((optic.xcount, 2), dtype = 'float')
        optic.cal_i()
        intensity[:, 0] = optic.xtick
        intensity[:, 1] = np.sum(optic.intensity, 0)
        
        np.savetxt(name, intensity)
        
    if t == 'mode':
        
        for i in range(c):
        
            name = optic.name + '_' + t + str(i) + '.dat'
            
            mode = np.zeros((n, 3), dtype = float)
            mode[:, 0] = x * 1e6
            mode[:, 1] = y * 1e6
            mode[:, 2] = np.abs(np.reshape(optic.cmode[i], (n, 1))[:, 0])**2
            mode[:, 2] = mode[:, 2] / np.max(mode[:, 2])
            
            np.savetxt(name, mode)
            
    if t == 'csd':
        
        name2dx = optic.name + '_' + 'csd2dx.dat'
        name2dy = optic.name + '_' + 'csd2dy.dat'
        name1dx = optic.name + '_' + 'csd1dx.dat'
        name1dy = optic.name + '_' + 'csd1dy.dat'
        
        c2dx = np.zeros((n, 3), dtype = float)
        c2dx[:, 0] = x
        c2dx[:, 1] = y
        c2dx[:, 2] = np.abs(np.reshape(optic.csd2dx, (n, 1))[:, 0])
        
        c2dy = np.zeros((n, 3), dtype = float)
        c2dy[:, 0] = x
        c2dy[:, 1] = y
        c2dy[:, 2] = np.abs(np.reshape(optic.csd2dy, (n, 1))[:, 0])
        
        c1dx = np.zeros((optic.xcount, 2), dtype = float)
        c1dx[:, 0] = optic.xtick
        c1dx[:, 1] = optic.csd1dx
        
        c1dy = np.zeros((optic.ycount, 2), dtype = float)
        c1dy[:, 0] = optic.ytick
        c1dy[:, 1] = optic.csd1dy
        
        np.savetxt(name2dx, c2dx)
        np.savetxt(name2dy, c2dy)
        np.savetxt(name1dx, c1dx)
        np.savetxt(name1dy, c1dy)
    
    if t == 'ratio':
        
        name = optic.name + '_' + t + '.dat'
        
        ratio = np.zeros((len(optic.cmode), 2), dtype = 'float')
        ratio[:, 0] = np.arange(len(optic.cmode))
        ratio[:, 1] = (optic.ratio[0 : len(optic.cmode)]/
                       np.sum(optic.ratio[0 : len(optic.cmode)]))
        
        np.savetxt(name, ratio)

def merge_optics(frames):
    
    xmin = list()
    xmax = list()
    ymin = list()
    ymax = list()
    
    re_frames = list()
    
    for i in range(len(frames)):
        
        xmin.append(frames[i].xstart)
        xmax.append(frames[i].xend)
        ymin.append(frames[i].ystart)
        ymax.append(frames[i].yend)
    
    xmin = np.max(np.array(xmin))
    xmax = np.min(np.array(xmax))
    ymin = np.max(np.array(ymin))
    ymax = np.min(np.array(ymax))
    
    locx_min = _locate(frames[0].xtick, xmin)
    locx_max = _locate(frames[0].xtick, xmax)
    locy_min = _locate(frames[0].ytick, xmin)
    locy_max = _locate(frames[0].ytick, xmax)
    
    for k in range(len(frames)):
        
        frames[k].xstart = xmin
        frames[k].xend   = xmax
        frames[k].ystart = ymin
        frames[k].yend   = ymax
        frames[k].xcount = int((xmax - xmin)/frames[k].xpixel)
        frames[k].ycount = int((ymax - ymin)/frames[k].ypixel)
        frames[k].xtick  = frames[k].xtick[locx_min : locx_max]
        frames[k].xtick  = frames[k].xtick[locx_min : locx_max]
        frames[k].ytick  = frames[k].ytick[locy_min : locy_max]
        frames[k].ytick  = frames[k].ytick[locy_min : locy_max]
        frames[k].gridx, frames[k].gridy = np.meshgrid(
            frames[k].xtick, frames[k].ytick
            )
        frames[k].mask = frames[k].mask[locx_min : locx_max, 
                                        locy_min : locy_max]
        
        for i in range(frames[k].n):
            frames[k].cmode[i] = frames[k].cmode[i][
                locx_min : locx_max, locy_min : locy_max
                ]
    
    return frames
