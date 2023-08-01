# PkBk

Computes Power Spectrum and Bispectrum multipoles using FFTs allowing for option of different Line-of-sights (LOS).

It can be used to quickly compute multipoles over many realisations with expansions for different LOS included as in ... . For the bispectrum several options are available but it can return values for the bispectrum over the full range of triangles for a given k-range.


It is intended as a fast user-friendly code written in python 3 with heavy use of numpy, and so nothing is needed to be compiled! The FFTs are implemented in c with pyfftw and other key bottlenecks are optimised with Numba. Mulithreading is implented through pyfftw and Numba and is used in some key areas. Multiproccessing can then be added on the frontend.



## Installation

git clone ....

Requirements:
- numpy
- scipy
- Numba
- pyfftw
- matplotlib (for frontend parts only)
- multiprocessing (for frontend parts only)

# Documentation and getting started

[View the documentation on Read the Docs](https://pkbk.readthedocs.io/en/latest/index.html)


See pk_example.ipynb and bk_example.ipynb for a quick start but a bit more thorough documentation is available down below.

Some extra information is avaliable in interpolating a box to get a over-density field.

## compute_grid_info.py

Contains functions which computes field information that only needs to be considered for a given box size and binning scheme:

Key functions:

### compute_survey

compute_survey(Nside,L,rfft=False,order=2,obs_pos=[0,0,0])

Computes key survey variables:

```
import compute_grid_info as cgi

N_side   = 128             #resolution of the grid
L        = 1000            #Length of side in cubic box in Mpc/h
rfft     = True            #whether to compute using real-to-complex Fourier transforms
order    = 2               #Order of Mass-assignment schement correction. 2 correspeonds to CIC (1- NGP, 3 TSC etc)
obs_pos  = [500,500,-500]  # [x,y,z] position of observer defined from origin in corner of box [Mpc/h]

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


### pk_compute_bins

pk_compute_bins(k,s,k_mag,k_f)

Computes binning scheme for the power spectrum and number of modes in each bin:

```
s=1/2 # bin width in units of 2*k_f
k_est = np.arange(grid_info[5],grid_info[6],2*s*grid_info[5]) + s*k_f #create k bins centers - from k_f to k_ny with steps of s*k_f
binning_info = cgi.pk_compute_bins(k_est,grid_info[3],grid_info[5],s)
In_bin,N_modes = binning_info
```
Outputs:
- **In_bin**: (N_bins,N_side,N_side,N_side) boolean array of whether each k_values is in each bin
- **N_modes**: Number of k-modes in each bin

---
### bk_full_compute_bins

bk_full_compute_bins(ks,N_side,s,k_mag,k_f,dtype=np.complex64,threads=1,rfft=False)

Computes binning scheme for the power spectrum and number of triangles in each bin via iFFTs:

```
ks       = np.arange(k_f,0.1,k_f)+ k_f/2  # k bins 
N_side   = 128                            # resolution of the grid
s        = 1/2 #units of 2*k_f            # bin width in units of 2*k_f
dtype    = np.complex64                   # dtype of fourier space array - either single of double precision
threads  = 6                              # Number of threads to use in FFTs
rfft     = True                           # whether to compute using real-to-complex Fourier transforms

binning_info = cgi.bk_full_compute_bins(ks,N_side,s,grid_info[3],grid_info[5],dtype,threads,rfft)
In_bin,N_tri = binning_info
```
Outputs:
- **In_bin**: (N_bins,N_side,N_side,N_side) boolean array of whether each k_values is in each bin
- **N_tri**: Number of triangles for each permutation of k1,k2,k3 bins


## Pk.py

- *delta*    The overdensity field in format of np.array((N_side, N_side, N_side), dtype=dtype_r)


## Bk_full.py


# Creative licence

# Cite


