# PkBk

Computes Power Spectrum and Bispectrum multipoles allowing for option of different Line-of-sights

## Installation

Requirements:
- numpy
- scipy
- Numba
- pyfftw

## Usage










After getting field...

### compute_grid_info.py

Contains functions which computes field information that only needs to be considered for a given box size and binning scheme:


**compute_survey**
Computes key survey variables:


- *N_side*   Resolution of the grid
- *L*        Length of side in cubic box
- *rfft*     

```
import compute_grid_info as cgi

N_side   = 128             #resolution of the grid
L        = 1000            #Length of side in cubic box in Mpc/h
rfft     = True            #whether to compute using real-to-complex Fourier transforms
order    = 2               #Order of Mass-assignment schement correction. 2 correspeonds to CIC (1- NGP, 3 TSC etc)
obs_pos  = [500,500,-500]  # [x,y,z] Position of observer defined from origin in corner of box [Mpc/h]

grid_info = cgi.compute_survey(N_side,L,rfft,order,obs_pos)
xi,x_norm,ki,k_mag,MAS,k_f,k_ny = grid_info
```
Outputs:
- *xi*: (3,N_side,N_side,N_side) grid of x,y,z positions of grid points defined from observer
- *x_norm*: Scalar distances of grid points from the observer
- *ki*: Grid of k-vectors
- *k_mag*: Grid of magnitude of k-vectors
- *MAS*: Mass assignment scheme correction to fourier space field
- *k_f*: Fundamental Frequency of box
- *k_ny*: Nyquist frequency of box






- *delta*    The overdensity field in format of np.array((N_side, N_side, N_side), dtype=dtype_r)

                 #field
L                     # 
Nside                 # grid resolution
k                     # k values to create bins at
grid_info             # output from grid_info func which contains information on the grid
binning_info          # output from binning_info func which contains information on the bins
t                     #as in Eq.
iFFT                  # use inverse fourier transforms to integral rather than simple sum - just keep it false!
dtype                 # single precision should be find and make it ~2x faster
real                  # use real-to-complex FFTs do cut down our k-space grid by half - save memory and fine for real power spectrum but not for odd l
verbose               # just print some extra info
```
In_bin,N_modes = pk_compute_bins(k,k_mag,k_f,s)# get binning information
```
### Power Spectrum