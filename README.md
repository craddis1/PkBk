# PkBk

Computes Power Spectrum and Bispectrum multipoles allowing for option of different Line-of-sights

## Installation



Requirements:
- numpy
- scipy
- NUMBA
- pyfftw

## Usage


### compute_grid_info.py

Computes field information that only needs to be considered for a given box size and resolution...


'xi,x_norm,ki,k_mag,MAS,k_f,k_ny = compute_survey(Nside,L,rfft,order,obs_pos) #get box variables

In_bin,N_modes = pk_compute_bins(k,k_mag,k_f,s)# get binning information'



 
