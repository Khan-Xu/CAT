# CAT
Coherence Analysis Toolbox

## enviroment

### python version: 3.7
### modules 
- numpy 1.18.4 
- mpi4py 3.0.3 
- matplotlib 3.2.1 
- scipy 1.4.1 
- h5py 2.10.0 
- srwpy 0.0.4
- pandas 1.0.1
- 
### workstation 
- CPU Intel Xeon Gold 6226 12C/24T
- RAM 64 G

## examples
### source calculation
at workstation, mpiexec -n 20 python HXCS_source_calculation.py

### others (use spyder or vscode, so that the results can be shown directly).
- plot coherent modes of source, HXCS_source.py 
- plot secondary source, HXCS_secondary_source.py
- plot coherent modes of X-ray beams at the sample plane, HXCS_sample_plane.py
- perform the simulation of CXDI, HXCS_CXDI.py
- perfrom the comparsion of propagtors, Test_multi_propagators_focus.py, Test_multi_propagators_diffract.py

Any question, please contact xuhan@ihep.ac.cn
citation：http://arxiv.org/abs/2110.09655
