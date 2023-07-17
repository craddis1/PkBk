# PkBk

Computes Power Spectrum and Bispectrum multipoles allowing for option of different Line-of-sights

## Installation



Requirements:
- numpy
- scipy
- NUMBA
- pyfftw

## Usage

### Power Spectrum








After getting field...

### compute_grid_info.py

Contains function which computes field information that only needs to be considered for a given box size and binning scheme:



Get box variables
```xi,x_norm,ki,k_mag,MAS,k_f,k_ny = compute_survey(Nside,L,rfft,order,obs_pos) #

In_bin,N_modes = pk_compute_bins(k,k_mag,k_f,s)# get binning information```


delta                 #field
L                     # boxsize
Nside                 # grid resolution
k                     #k values to create bins at
grid_info             # output from grid_info func which contains information on the grid
binning_info          # output from binning_info func which contains information on the bins
t                     #as in Eq.
iFFT                  # use inverse fourier transforms to integral rather than simple sum - just keep it false!
dtype                 # single precision should be find and make it ~2x faster
real                  # use real-to-complex FFTs do cut down our k-space grid by half - save memory and fine for real power spectrum but not for odd l
verbose               # just print some extra info
 
