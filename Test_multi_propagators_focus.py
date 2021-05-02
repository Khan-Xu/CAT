#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"

#------------------------------------------------------------------------------
# modules

try:
    from srwpy.srwlib import *
    from srwpy.uti_plot import *
except:
    from oasys_srw.srwlib import *
    from oasys_srw.uti_plot import *

if not srwl_uti_proc_is_master(): exit()

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from cat.optics import source, source2, ideal_lens, KB, screen, slit, AKB
from cat.propagate import propagate_s, propagate_k, propagate_czt
from cat import tool

#------------------------------------------------------------------------------
# functions

def srw_cat_compare_propagators():

    #--------------------------------------------------------------------------
    # wavefront calculation
    
    part_beam = SRWLPartBeam()
    part_beam.Iavg               = 0.2
    part_beam.partStatMom1.x     = 0.0
    part_beam.partStatMom1.y     = 0.0
    part_beam.partStatMom1.z     = -2.0795500000000002
    part_beam.partStatMom1.xp    = 0.0
    part_beam.partStatMom1.yp    = 0.0
    part_beam.partStatMom1.gamma = 11741.70710144324
    part_beam.arStatMom2[0]      = 8.704890000000001e-11
    part_beam.arStatMom2[1]      = 0.0
    part_beam.arStatMom2[2]      = 1.1102224e-11
    part_beam.arStatMom2[3]      = 5.938969e-12
    part_beam.arStatMom2[4]      = 0.0
    part_beam.arStatMom2[5]      = 1.6281759999999997e-12
    part_beam.arStatMom2[10]     = 1.1236e-06
    
    magnetic_fields = []
    magnetic_fields.append(SRWLMagFldH(
        1, 'v', 
        _B=0.47251184326559015, 
        _ph=0.0, 
        _s=-1, 
        _a=1.0)
        )
    magnetic_structure = SRWLMagFldU(
        _arHarm=magnetic_fields, _per=0.0199, _nPer=201.0
        )
    magnetic_field_container = SRWLMagFldC(
        _arMagFld=[magnetic_structure], 
        _arXc=array('d', [0.0]), 
        _arYc=array('d', [0.0]), 
        _arZc=array('d', [0.0])
        )
    
    mesh = SRWLRadMesh(
        _eStart=12400.003208235006, _eFin=12400.003208235006,
        _ne    =1,
        _xStart=-0.0002, _xFin=0.0002, _nx=513,
        _yStart=-0.0002, _yFin=0.0002, _ny=513,
        _zStart=10.0
        )
    
    stk = SRWLStokes()
    stk.allocate(1,513,513)
    stk.mesh = mesh
    
    wfr = SRWLWfr()
    wfr.allocate(mesh.ne, mesh.nx, mesh.ny)
    wfr.mesh = mesh
    wfr.partBeam = part_beam
    
    initial_mesh = deepcopy(wfr.mesh)
    srwl.CalcElecFieldSR(
        wfr, 0, magnetic_field_container, [1,0.01,0.0,0.0,50000,1,0.0]
        )
    
    # example wfr
    
    mesh0 = deepcopy(wfr.mesh)
    wfr0 = np.reshape(np.array(wfr.arEx), (mesh0.nx, mesh0.ny, 2))
    wfr0 = wfr0[:, :, 0] + 1j*wfr0[:, :, 1]
    
    # plot wfr
    
    arI = array('f', [0]*mesh0.nx*mesh0.ny)
    srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, mesh0.eStart, 0, 0)
    plotMesh0x = [1000*mesh0.xStart, 1000*mesh0.xFin, mesh0.nx]
    plotMesh0y = [1000*mesh0.yStart, 1000*mesh0.yFin, mesh0.ny]
    
    arI = np.reshape(np.array(arI), (mesh0.nx, mesh0.ny))
    xtick = np.linspace(mesh0.xStart*1e6, mesh0.xFin*1e6, mesh0.nx)
    ytick = np.linspace(mesh0.yStart*1e6, mesh0.yFin*1e6, mesh0.ny)
    
    #--------------------------------------------------------------------------
    # optical layout
    
    sr0 = source2(file_name = 'b4_srw2_12400.h5', name = 'source', n_vector = 1)
    sr0.slit(xcoor = [-2e-4, 2e-4], ycoor = [-2e-4, 2e-4])
    sr0.remap(0.77973e-6, 0.77973e-6)
    sr0.cmode[0] = np.abs(wfr0) * np.exp(-1j * np.angle(wfr0))
    sr0.position = 10
    
    crl = ideal_lens(optics = sr0, n = 1, location = 10,
                     xfocus = 5, yfocus = 5)
    propagate_s(sr0, crl)
    crl.remap(0.3e-6, 0.3e-6)
    
    sc0 = screen(optics = crl, n = 1, location = 20)
    sc1 = screen(optics = crl, n = 1, location = 20)
    sc2 = screen(optics = crl, n = 1, location = 20)
    
    #--------------------------------------------------------------------------
    # compare propagators
    
    # Fresnel propagator
    propagate_s(crl, sc0, t = 'fresnel')
    
    # Asm propagator
    propagate_s(crl, sc1, t = 'asm')
    
    # CZT propagator
    propagate_czt(crl, sc2)
    
    #--------------------------------------------------------------------------
    
    return [xtick, ytick, arI], sc0, sc1, sc2

def srw_foucs():
    
    
    #--------------------------------------------------------------------------
    # beamline
    
    part_beam = SRWLPartBeam()
    part_beam.Iavg               = 0.2
    part_beam.partStatMom1.x     = 0.0
    part_beam.partStatMom1.y     = 0.0
    part_beam.partStatMom1.z     = -2.0795500000000002
    part_beam.partStatMom1.xp    = 0.0
    part_beam.partStatMom1.yp    = 0.0
    part_beam.partStatMom1.gamma = 11741.70710144324
    part_beam.arStatMom2[0]      = 8.704890000000001e-11
    part_beam.arStatMom2[1]      = 0.0
    part_beam.arStatMom2[2]      = 1.1102224e-11
    part_beam.arStatMom2[3]      = 5.938969e-12
    part_beam.arStatMom2[4]      = 0.0
    part_beam.arStatMom2[5]      = 1.6281759999999997e-12
    part_beam.arStatMom2[10]     = 1.1236e-06
    
    magnetic_fields = []
    magnetic_fields.append(SRWLMagFldH(1, 'v', 
                                       _B=0.47251184326559015, 
                                       _ph=0.0, 
                                       _s=-1, 
                                       _a=1.0))
    magnetic_structure = SRWLMagFldU(_arHarm=magnetic_fields, _per=0.0199, _nPer=201.0)
    magnetic_field_container = SRWLMagFldC(_arMagFld=[magnetic_structure], 
                                           _arXc=array('d', [0.0]), 
                                           _arYc=array('d', [0.0]), 
                                           _arZc=array('d', [0.0]))
    
    mesh = SRWLRadMesh(_eStart=12400.003208235006,
                       _eFin  =12400.003208235006,
                       _ne    =1,
                       _xStart=-0.0002,
                       _xFin  =0.0002,
                       _nx    =512,
                       _yStart=-0.0002,
                       _yFin  =0.0002,
                       _ny    =512,
                       _zStart=10.0)
    
    stk = SRWLStokes()
    stk.allocate(1,512,512)
    stk.mesh = mesh
    
    wfr = SRWLWfr()
    wfr.allocate(mesh.ne, mesh.nx, mesh.ny)
    wfr.mesh = mesh
    wfr.partBeam = part_beam
    
    initial_mesh = deepcopy(wfr.mesh)
    srwl.CalcElecFieldSR(wfr, 0, magnetic_field_container, [1,0.01,0.0,0.0,50000,1,0.0])
    
    mesh0 = deepcopy(wfr.mesh)
    arI = array('f', [0]*mesh0.nx*mesh0.ny)
    srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, mesh0.eStart, 0, 0)
    arIx = array('f', [0]*mesh0.nx)
    srwl.CalcIntFromElecField(arIx, wfr, 6, 0, 1, mesh0.eStart, 0, 0)
    arIy = array('f', [0]*mesh0.ny)
    srwl.CalcIntFromElecField(arIy, wfr, 6, 0, 2, mesh0.eStart, 0, 0)
    
    #--------------------------------------------------------------------------
    # beamline
    
    srw_oe_array = []
    srw_pp_array = []
    
    oe_0=SRWLOptL(_Fx=5.0, _Fy=5.0, _x=0.0, _y=0.0)
    
    pp_oe_0 = [0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0]
    
    srw_oe_array.append(oe_0)
    srw_pp_array.append(pp_oe_0)
    
    drift_after_oe_0 = SRWLOptD(10.0)
    pp_drift_after_oe_0 = [0,0,1.0,1,0,1.0,1.0,1.0,1.0,0,0.0,0.0]
    
    srw_oe_array.append(drift_after_oe_0)
    srw_pp_array.append(pp_drift_after_oe_0)
    
    
    #--------------------------------------------------------------------------
    # propagation
    
    import numpy as np
    
    optBL = SRWLOptC(srw_oe_array, srw_pp_array)
    srwl.PropagElecField(wfr, optBL)
    
    mesh1 = deepcopy(wfr.mesh)
    arI1 = array('f', [0]*mesh1.nx*mesh1.ny)
    srwl.CalcIntFromElecField(arI1, wfr, 6, 0, 3, mesh1.eStart, 0, 0)
    
    intensity = np.reshape(np.array(arI1), (mesh1.nx, mesh1.ny))
    xtick = np.linspace(1e3*mesh1.xStart, 1e3*mesh1.xFin, mesh1.nx)
    ytick = np.linspace(1e3*mesh1.yStart, 1e3*mesh1.yFin, mesh1.ny)
    
    #--------------------------------------------------------------------------
    
    return xtick, ytick, intensity

if __name__ == '__main__':
    
    # srw propagation
    
    srw_xtick, srw_ytick, intensity = srw_foucs()
    wfr0, sc0, sc1, sc2 = srw_cat_compare_propagators()
    
    #--------------------------------------------------------------------------
    # 2d plot
    
    # wfr0
    plt.figure(figsize = (6, 6))
    plt.pcolor(wfr0[0], wfr0[1], wfr0[2])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # focus
    
    plt.figure(figsize = (6, 6))
    plt.pcolor(srw_xtick[183 : 328]*1e3, srw_ytick[183 : 328]*1e3, intensity[183 : 328, 183 : 328])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.figure(figsize=(6, 6))
    sc0.slit(xcoor = [-5e-5, 5e-5], ycoor = [-5e-5, 5e-5])
    plt.pcolor(sc0.xtick*1e6, sc0.ytick*1e6, np.abs(sc0.cmode[0])**2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.figure(figsize=(6, 6))
    xshift = -2.529 # um
    sc1.slit(xcoor = [-5e-5, 5e-5], ycoor = [-5e-5, 5e-5])
    plt.pcolor(sc1.xtick*1e6, sc1.ytick*1e6, np.abs(sc1.cmode[0])**2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.figure(figsize=(6, 6))
    xshift = -0.289
    sc2.slit(xcoor = [-5e-5, 5e-5], ycoor = [-5e-5, 5e-5])
    plt.pcolor(sc2.xtick*1e6, sc2.ytick*1e6, np.abs(sc2.cmode[0])**2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    #--------------------------------------------------------------------------
    # 1d plot
    
    plt.figure(figsize = (6, 6))
    
    plt.plot(srw_xtick[183 : 328]*1e3, np.sum(intensity[183 : 328, 183 : 328], 0) / np.max(np.sum(intensity[183 : 328, 183 : 328], 0)))
    plt.plot(sc0.xtick*1e6, np.sum(np.abs(sc0.cmode[0])**2, 0) / np.max(np.sum(np.abs(sc0.cmode[0])**2, 0)))
    plt.plot(sc0.xtick*1e6 + 2.529, np.sum(np.abs(sc1.cmode[0])**2, 0) / np.max(np.sum(np.abs(sc1.cmode[0])**2, 0)))
    plt.plot(sc0.xtick*1e6 + 0.289, np.sum(np.abs(sc2.cmode[0])**2, 0) / np.max(np.sum(np.abs(sc2.cmode[0])**2, 0)))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)