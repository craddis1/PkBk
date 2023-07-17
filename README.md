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

Key functions:
**compute_survey**
Computes key survey variables:

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
- **xi**: (3,N_side,N_side,N_side) grid of x,y,z positions of grid points defined from observer
- **x_norm**: Scalar distances of grid points from the observer
- **ki**: Grid of k-vectors
- **k_mag**: Grid of magnitude of k-vectors
- **MAS**: Mass assignment scheme correction to fourier space field
- **k_f**: Fundamental Frequency of box
- **k_ny**: Nyquist frequency of box

---
**pk_compute_bins**
Computes binning scheme for the power spectrum and number of modes in each bin:

```
s=1/2 # bin width in units of 2*k_f
k_est = np.arange(grid_info[5],grid_info[6],2*s*grid_info[5]) + s*k_f #create k bins centers - from k_f to k_ny with steps of s*k_f
binning_info = pk_compute_bins(k_est,grid_info[3],grid_info[5],s)
In_bin,N_modes = binning_info
```
Outputs:
- **In_bin**: (N_bins,N_side,N_side,N_side) boolean array of whether each k_values is in each bin
- **N_modes**: Number of k-modes in each bin



- *delta*    The overdensity field in format of np.array((N_side, N_side, N_side), dtype=dtype_r)


### Power Spectrum