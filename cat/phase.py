

"""
_phase: phase tools for Coherent imaging method.
 
Functions: None.      

Classes  : None.
"""

#-----------------------------------------------------------------------------#
# library

import numpy as np
import scipy as sp

#-----------------------------------------------------------------------------#
# function

def siemens_star(dsize = 4096, nb_rays = 72, r_max = 2000, nb_rings = 20, 
                 cheese_holes_nb = 0, cheese_hole_max_radius = 10,
                 cheese_hole_spiral_period = 0):

    """
    This function is copyed from pynx. Set a sample for experiment.

    origal comment from pynx:
    
    Calculate a binary Siemens star.

    Args:
        dsize:                     size in pixels for the 2D array with the 
                                   star data.
        nb_rays:                   number of radial branches for the star. 
                                   Must be > 0
        r_max:                     maximum radius for the star in pixels. 
                                   If None, dsize/2 is used.
        nb_rings:                  number of rings (the rays will have some 
                                   holes between successive rings).
        cheese_holes_nb:           number of cheese holes other the entire 
                                   area, resulting more varied frequencies.
        cheese_hole_max_radius:    maximum axial radius for the holes (with 
                                   random radius along x and y). If the value 
                                   is negative, the radius is fixed instead of 
                                   random.
        cheese_hole_spiral_period: instead of randomly distributing holes, 
                                   giving an integer N number for this parameter
                                   will generate holes located on an Archimedes 
                                   spiral, every N pixels along the spiral, 
                                   which also has a step of N pixels. The 
                                   pattern of holes is aperiodic.

    Returns:
        a 2D array with the Siemens star.
    """
    
    def spiral_archimedes(a, n):
        
        """" 
        Creates np points spiral of step a, with a between successive points
        on the spiral. Returns the x,y coordinates of the spiral points.
    
        This is an Archimedes spiral. the equation is:
          r = (a/2*pi) * theta
          the stepsize (radial distance between successive passes) is a
          the curved absciss is: 
              s(theta) = (a/2*pi) * integral[t=0->theta](sqrt(1*t**2))dt
        """
        
        vr, vt = [0], [0]
        t = np.pi
        
        while len(vr) < n:
            
            vt.append(t)
            vr.append(a * t / (2 * np.pi))
            t += 2 * np.pi / np.sqrt(1 + t ** 2)
        
        vt, vr = np.array(vt), np.array(vr)
        
        return vr * np.cos(vt), vr * np.sin(vt)


    if r_max is None:
        r_max = dsize // 2
        
    x, y = np.meshgrid(np.arange(-dsize // 2, dsize // 2, dtype=np.float32),
                       np.arange(-dsize // 2, dsize // 2, dtype=np.float32))

    a = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    am = 2 * np.pi / nb_rays
    d = (a % (am)) < (am / 2)
    
    if r_max != 0 and r_max is not None:
        d *= r < r_max
    
    if nb_rings != 0 and nb_rings is not None:
        if r_max is None:
            rm = dsize * np.sqrt(2) / 2 / nb_rings
        else:
            rm = r_max / nb_rings
        d *= (r % rm) < (rm * 0.9)
    
    if cheese_holes_nb > 0:
       
        if cheese_hole_spiral_period:
            cx, cy = spiral_archimedes(cheese_hole_spiral_period, 
                                       cheese_holes_nb + 1)
            # remove center
            cx = cx[1:].astype(np.int32)
            cy = cy[1:].astype(np.int32)
        
        else:
            cx = np.random.randint(x.min(), x.max(), cheese_holes_nb)
            cy = np.random.randint(y.min(), y.max(), cheese_holes_nb)
        
        if cheese_hole_max_radius < 0:
            rx = (np.ones(cheese_holes_nb) * 
                  abs(cheese_hole_max_radius)).astype(np.int32)
            ry = rx
        
        else:
            rx = np.random.uniform(1, cheese_hole_max_radius, cheese_holes_nb)
            ry = np.random.uniform(1, cheese_hole_max_radius, cheese_holes_nb)
        
        for i in range(cheese_holes_nb):
            
            dn = int(np.ceil(max(rx[i], ry[i])))
            x0, x1 = dsize // 2 + cx[i] - dn, dsize // 2 + cx[i] + dn
            y0, y1 = dsize // 2 + cy[i] - dn, dsize // 2 + cy[i] + dn
            
            d[y0 : y1, x0 : x1] *= (
                ((x[y0:y1, x0:x1] - cx[i]) / rx[i]) ** 2 + 
                ((y[y0:y1, x0:x1] - cy[i]) / ry[i]) ** 2
                ) > 1
    
    return d.astype(np.float32)

def hio_er(intensity, support, loop_hio = 5000, loop_er = 500, iteration = 1,
           beta = 0.9, real = 1, show = 0, internal = 0):

    """
    CDI : hio + er methods to reonstruct the phase of diffraction.

    Args:
        intensity:  diffraction intensity (not amplitude).
        support:    inital support of sample.
        loop_hio:   loop number of hio.
        loop_er:    loop number of er.
        beta:       beta parameters of hio.

    Returns:
        errors:     square errors during the loop.
        r1_space:   reconstructed reusult.
    """
    
    shape = intensity.shape
    
    # phase initialization
    
    diff = np.sqrt(intensity)
    phase0 = np.random.rand(shape[0], shape[1])
    
    # convex random phase.
    
    phase0 = np.angle(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(phase0))))
    q0_space = diff * np.exp(1j * phase0)
    r0_space = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(q0_space))).real
    # r0_space = r0_space.real
    # iteration
    
    if internal:
        frames = list()
    
    errors = list()
    
    for k in range(int(iteration)):
        
        # hio process
        
        for h in range(int(loop_hio)):
        
            r1_space = np.fft.ifftshift(
                np.fft.ifft2(np.fft.ifftshift(q0_space))
                ).real
            # r1_space = r1_space.real
            # data replacement
            
            sample = r1_space * support
            # sample[sample < 0] = 0
            
            # hio constrain
            
            r1_space = r0_space - beta * r1_space
            r1_space[support == 1] = sample[support == 1]
           
            # cal new phase.
            
            r0_space = r1_space
            q1_space = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(r1_space)))
            p_angle = np.angle(q1_space)
            
            # cal error
            
            r2 = np.sum(
                abs(diff - 
                    np.abs(
                        np.fft.fftshift(np.fft.fft2(np.fft.fftshift(sample)))
                        ))
                ) / np.sum(diff) 
            errors.append(r2)
            
            if show : print([h, r2], flush = True)
            
            # update new phase  
            
            q0_space = diff * np.exp(1j *p_angle)
            
            if internal:
                if h % 10 == 0:
                    frames.append(np.abs(r1_space))
            
        # er process
            
        for e in range(int(loop_er)):
            
            r1_space = np.fft.ifftshift(
                np.fft.ifft2(np.fft.ifftshift(q0_space))
                ).real
            
            # data replacement
            
            sample = r1_space * support
            # sample[sample < 0] = 0
            
            # hio constrain
            
            r1_space = 0.00 * r0_space
            r1_space[support == 1] = sample[support == 1]
            
            # cal new phase.
            
            r0_space = r1_space
            q1_space = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(r1_space)))
            p_angle = np.angle(q1_space)
            
            # cal error
            
            r2 = np.sum(
                abs(diff - 
                    np.abs(
                        np.fft.fftshift(np.fft.fft2(np.fft.fftshift(sample)))
                        ))
                ) / np.sum(diff) 
            errors.append(r2)
            
            if show : print([e, r2], flush = True)
            
            # update new phase  
            
            q0_space = diff * np.exp(1j *p_angle)
    
    if internal:
        return errors, r1_space, frames
    else:
        return errors, r1_space
            
            
def hio_er2(intensity, support, loop_hio = 5000, loop_er = 500, iteration = 1,
           beta = 0.9, real = 1, show = 0, internal = 0):

    """
    CDI : hio + er methods to reonstruct the phase of diffraction.

    Args:
        intensity:  diffraction intensity (not amplitude).
        support:    inital support of sample.
        loop_hio:   loop number of hio.
        loop_er:    loop number of er.
        beta:       beta parameters of hio.

    Returns:
        errors:     square errors during the loop.
        r1_space:   reconstructed reusult.
    """
    
    shape = intensity.shape
    
    # phase initialization
    
    diff = np.sqrt(intensity)
    phase0 = np.random.rand(shape[0], shape[1])
    
    # convex random phase.
    
    phase0 = np.angle(sp.fft.ifftshift(sp.fft.ifft2(sp.fft.ifftshift(phase0))))
    q0_space = diff * np.exp(1j * phase0)
    r0_space = sp.fft.ifftshift(sp.fft.ifft2(sp.fft.ifftshift(q0_space))).real
    # r0_space = r0_space.real
    # iteration
    
    if internal:
        frames = list()
    
    errors = list()
    
    for k in range(int(iteration)):
        
        # hio process
        
        for h in range(int(loop_hio)):
        
            r1_space = sp.fft.ifftshift(
                sp.fft.ifft2(sp.fft.ifftshift(q0_space))
                ).real
            # r1_space = r1_space.real
            # data replacement
            
            sample = r1_space * support
            # sample[sample < 0] = 0
            
            # hio constrain
            
            r1_space = r0_space - beta * r1_space
            r1_space[support == 1] = sample[support == 1]
           
            # cal new phase.
            
            r0_space = r1_space
            q1_space = sp.fft.fftshift(sp.fft.fft2(sp.fft.ifftshift(r1_space)))
            p_angle = np.angle(q1_space)
            
            # cal error
            
            r2 = np.sum(
                abs(diff - 
                    np.abs(
                        sp.fft.fftshift(sp.fft.fft2(sp.fft.fftshift(sample)))
                        ))
                ) / np.sum(diff) 
            errors.append(r2)
            
            if show : print([h, r2], flush = True)
            
            # update new phase  
            
            q0_space = diff * np.exp(1j *p_angle)
            
            if internal:
                if h % 10 == 0:
                    frames.append(np.abs(r1_space))
            
        # er process
            
        for e in range(int(loop_er)):
            
            r1_space = sp.fft.ifftshift(
                sp.fft.ifft2(sp.fft.ifftshift(q0_space))
                ).real
            
            # data replacement
            
            sample = r1_space * support
            # sample[sample < 0] = 0
            
            # hio constrain
            
            r1_space = 0.00 * r0_space
            r1_space[support == 1] = sample[support == 1]
            
            # cal new phase.
            
            r0_space = r1_space
            q1_space = sp.fft.fftshift(sp.fft.fft2(sp.fft.ifftshift(r1_space)))
            p_angle = np.angle(q1_space)
            
            # cal error
            
            r2 = np.sum(
                abs(diff - 
                    np.abs(
                        sp.fft.fftshift(sp.fft.fft2(sp.fft.fftshift(sample)))
                        ))
                ) / np.sum(diff) 
            errors.append(r2)
            
            if show : print([e, r2], flush = True)
            
            # update new phase  
            
            q0_space = diff * np.exp(1j *p_angle)
    
    if internal:
        return errors, r1_space, frames
    else:
        return errors, r1_space            
                
    
    