"""
Computes field information - on relvant k-vectors and k-magnitudes as well as correcting for MAS scheme
NUMBA adds something but not much considering this will be called when changing the grid parameters

xi,ki,k_mag,MAS,k_f = compute_survey(Nside,L,rfft=False,order=2,obs_pos=[0,0,0])

In_bin,N_modes = pk_compute_bins(k,k_mag,k_f,s)

or 
In_bin,Ntri,In_bin1 = bk_equalateral_compute_bins(k_eq,N_side,s,k_mag,k_f,iso_f = 1,dtype=np.complex64,threads=1,rfft=False)
#then we have bispectrum precompute stuff for sum and iFFT cases...
need iFFTs
"""
from numba import jit, complex64, complex128,prange

import numpy as np
import sys
sys.path.append('Library')

@jit(nopython=True)#if we dont use NUMBA - then can use np.meshgrid
def meshgrid(x, y, z):  #meshgrid compatible with NUMBA
    xx = np.empty(shape=(x.size, y.size, z.size))
    yy = np.empty(shape=(x.size, y.size, z.size))
    zz = np.empty(shape=(x.size, y.size, z.size))
    for i in range(x.size):
        for j in range(y.size):
            for k in range(z.size):
                xx[i,j,k] = x[i]  # this indexing with this order returns ij indexing
                yy[i,j,k] = y[j]  #
                zz[i,j,k] = z[k]
    return xx, yy, zz

@jit(nopython=True) #~30% speed up with NUMBA
def MAS_corr_field(k,k_r,k_ny,order): #correction for cic
    x = k/(2*k_ny);x_r = k_r/(2*k_ny)   #for both normal and real options
    corr = (np.sinc(x))**(-order);corr_r = (np.sinc(x_r))**(-order)
    #rebuild meshgrid
    corr_grid = np.expand_dims(corr,axis=(1))*corr    #expand to (Nside,Nside)          #corr_grid = corr[:,np.newaxis]*corr
    corr_field = np.expand_dims(corr_grid,axis=(2))*corr_r        #corr_field = corr_grid[:,:,np.newaxis]*corr
    return corr_field


#creates LoS to convolve with the field for Qpqrs - gets n_hat direction of LOS vector
@jit(nopython=True) #njitted version with custom meshgrid!!! ~10-20% quicker from NUMBA
def LoS(Nside,L,obs_pos):
    conf_space = np.linspace(0,L,Nside)  #CHANGE - default is to define coordinates from the box centre
    x , y , z = meshgrid(conf_space- L/2-obs_pos[0], conf_space- L/2-obs_pos[1], conf_space- L/2-obs_pos[2]) 
    conf_norm = np.sqrt(x**2 + y**2 + z**2) # make a unit vector - normalise
    
    #avoid zero errors:
    conf_norm = np.where(conf_norm==0,1e-10,conf_norm)                     #where conf_norm is 0, - so is x,y,z!!!
    xi = np.empty((3,Nside,Nside,Nside))
    xi[0]=x;xi[1]=y;xi[2]=z               #i.e x1
    return xi,conf_norm

@jit(nopython=True)
def get_field_variables(Nside,Nside_r,L,k_modes,k_modes_r,order):      # gets data and constants for box Nside,L
    kx , ky , kz = meshgrid(k_modes,k_modes, k_modes_r)

    k_mag = np.sqrt((kx**2+ky**2+kz**2))
    
    k_f = 2*np.pi/L  #fundamental mode of the box
    
    k_ny = k_f*Nside/2 #nyquist frequency
    #implements MAS correction #returns for all 3 directions
    MAS = MAS_corr_field(k_modes,k_modes_r,k_ny,order)
    
    #so we can avoid zero division errors
    k_mag0 = np.where(k_mag==0,1,k_mag)
    ki = np.empty((3,Nside,Nside,Nside_r))
    ki[0]=kx/k_mag0;ki[1]=ky/k_mag0;ki[2]=kz/k_mag0       # normalised k-vectors   
    
    return ki, k_mag, MAS, k_f, k_ny

#this is run once for each grid and then stored 
#once in grid and for each set of bins!
#@jit(forceobj=True) #no real speed up for NUMBA here but the is in the two funcs it calls
def compute_survey(Nside,L,rfft=False,order=2,obs_pos=[0,0,0]):
    k_modes = 2*np.pi*np.fft.fftfreq(Nside,d=(L/Nside)) # get frequencies in one direction !!! #has to be in object mode
    
    if rfft == True: #if rfft
        Nside_r = Nside//2 + 1
        k_modes_r = 2*np.pi*np.fft.rfftfreq(Nside,d=(L/Nside))
    else:
        k_modes_r = k_modes
        Nside_r = Nside

    ki, k_mag, MAS, k_f,k_ny = get_field_variables(Nside,Nside_r,L,k_modes,k_modes_r,order)
    xi,x_norm = LoS(Nside,L,obs_pos)# returns normalised postions of boxes configuration space
    return xi,x_norm,ki,k_mag,MAS,k_f,k_ny

#separate func just for binning info - computes which modes are in bin and sums for normalisation; as well as creating boolean mask
@jit(nopython=True) #good NUMBA gains -idk at least 50%
def pk_compute_bins(k,s,k_mag,k_f): #computes binning information for power spectrum #this is dependent on compute grid info
    N_modes = np.zeros(len(k))
    In_bin = np.zeros((len(k),*k_mag.shape),dtype=np.bool_)
    for i in range(len(k)): 
            In_bin[i] = np.abs(k[i] - k_mag)<s*k_f
            N_modes[i] = np.sum(In_bin[i])
            
    #removes divide by zero errors (pk will be zero for 0 Nmodes)
    N_modes = np.where(N_modes==0,1,N_modes)
    return In_bin,N_modes

@jit(nopython=True,parallel=True)#,fastmath=fastmath
def ifft_sum(field1: complex64[:, :], field2: complex64[:, :], field3: complex64[:, :]) -> complex64[:, :, :]: # does sum over fields where there are possible closed triangles...   
    N_bins= len(field1)
    Bk_lm = np.zeros((N_bins,N_bins,N_bins),dtype=np.complex64)
    for i in prange(N_bins):
        for j in range(N_bins):
            for k in range(N_bins):
                if (i > j+k+1) or ((j > i+k+1)or (k > i+j+1)):#these values should all be 0  as dirac delta
                    continue
                sum_real = 0.0
                sum_imag = 0.0
                for idx in range(field1.shape[1]):
                    sum_real += field1.real[i, idx] * field2.real[j, idx] * field3.real[k, idx] - field1.imag[i, idx] * field2.imag[j, idx] * field3.real[k, idx] - field1.real[i, idx] * field2.imag[j, idx] * field3.imag[k, idx] - field1.imag[i, idx] * field2.real[j, idx] * field3.imag[k, idx]
                    sum_imag += field1.real[i, idx] * field2.imag[j, idx] * field3.real[k, idx] + field1.imag[i, idx] * field2.real[j, idx] * field3.real[k, idx] + field1.real[i, idx] * field2.real[j, idx] * field3.imag[k, idx] - field1.imag[i, idx] * field2.imag[j, idx] * field3.imag[k, idx]
                Bk_lm[i, j, k] = sum_real + 1j*sum_imag

    return Bk_lm

#separate func just for binning info -
#theta binning scheme
#find the modes in bins for k1 k2 and k3 and then computes the number of closed triangles using iFFTs.
#@jit(nopython=True)
def bk_full_compute_bins(ks,N_side,s,k_mag,k_f,dtype=np.complex64,threads=1,rfft=False): #computes binning information for power spectrum #this is dependent on compute grid info
    #raise warning if bad dtype
    if dtype != np.complex128 and dtype != np.complex64:
        raise Exception("Invalid dtype")

    #import which fft type we want to use for the estimator
    #also sets some values in the case of real FFTs - i.e. fourier space arrays are halved
    if rfft:
        N_side_r = N_side//2 + 1                   #N_side changes in last axis for rffts
        if dtype == np.complex128:
            from FFTW import FFTW_irfft_double as iFFT_
            dtype_r = np.float64                            #dtype changes in real space for rffts
        elif dtype == np.complex64:
            from FFTW import FFTW_irfft_single as iFFT_
            dtype_r = np.float32
    else:
        if dtype == np.complex128:
            from FFTW import FFTW_ifft_double as iFFT_
        elif dtype == np.complex64:
            from FFTW import FFTW_ifft_single as iFFT_
        dtype_r = dtype                         
        N_side_r = N_side                       

    #wrapper function for iFFT scheme imported from FFTW module - counts iFFTs
    def FFTW_ifft(delta):
        #global iFFT_number
        #iFFT_number += 1
        return iFFT_(delta,threads)

    N_bins = len(ks)
    #set our number of bins for k3 from theta
    bins = np.zeros((N_bins,2))
    for i in range(N_bins):
        bins[i,:] = [ks[i]-s*k_f,ks[i]+s*k_f]
    
    #find boolean of inbin for all bins...
    In_bin = np.zeros((N_bins,*k_mag.shape),dtype=np.bool_)
    for i in range(N_bins): #for each bin a new FFT box is created to get Delta(n,k) and I(n,k) after being IFFTed
        In_bin[i] = np.where(np.logical_and(k_mag>bins[i,0],k_mag<bins[i,1]),True,False)#boolean type whether each k-vector is in bin
   
    def ifft_field(field):#lets edit this...
        N_bins= len(In_bin)
        N_side = field.shape[1]
        ifft_F = np.zeros((N_bins,N_side**3),dtype=dtype_r) #create empty arrays for the iffts of each bins
        for i in range(N_bins):#for each bin a new FFT box is created to get I(n,k) after being IFFTed
            ifft_F[i] = FFTW_ifft(field[i]).flatten()
        return ifft_F

    ifftbox = np.where(In_bin,1,0)
    fft_I = ifft_field(ifftbox)#inverse fourier transform boolean
        
    #now compute Ntri
    Ntri = ifft_sum(fft_I,fft_I,fft_I)

    #removes divide by zero errors (bk will be zero for 0 Ntri)
    Ntri = np.where(Ntri==0,1,Ntri)
    return In_bin,Ntri

#separate func just for binning info -
#theta binning scheme
#find the modes in bins for k1 k2 and k3 and then computes the number of closed triangles using iFFTs.
#@jit(nopython=True)
def bk_theta_compute_bins(k1,k2,theta,N_side,s,k_mag,k_f,dtype=np.complex64,threads=1,rfft=False): #computes binning information for power spectrum #this is dependent on compute gird info
        #raise warning if bad dtype
    if dtype != np.complex128 and dtype != np.complex64:
        raise Exception("Invalid dtype")

    #import which fft type we want to use for the estimator
    #also sets some values in the case of real FFTs - i.e. fourier space arrays are halved
    if rfft:
        N_side_r = N_side//2 + 1                   #N_side changes in last axis for rffts
        if dtype == np.complex128:
            from FFTW import FFTW_irfft_double as iFFT_
            dtype_r = np.float64                            #dtype changes in real space for rffts
        elif dtype == np.complex64:
            from FFTW import FFTW_irfft_single as iFFT_
            dtype_r = np.float32
    else:
        if dtype == np.complex128:
            from FFTW import FFTW_ifft_double as iFFT_
        elif dtype == np.complex64:
            from FFTW import FFTW_ifft_single as iFFT_
        dtype_r = dtype                         
        N_side_r = N_side                       

    #wrapper function for iFFT scheme imported from FFTW module - counts iFFTs
    def FFTW_ifft(delta):
        #global iFFT_number
        #iFFT_number += 1
        return iFFT_(delta,threads)
        
    k3 = np.sqrt((k2*np.sin(theta))**2 + (k2*np.cos(theta)+k1)**2) #get k3

    k1bin = [k1-s*k_f,k1+s*k_f] #maybe change to s/2
    k2bin = [k2-s*k_f,k2+s*k_f]

    in_k1 = np.logical_and(k_mag>k1bin[0],k_mag<k1bin[1]) # defines k-vectors in bin
    in_k2 = np.logical_and(k_mag>k2bin[0],k_mag<k2bin[1]) # defines k-vectors in bin

    fft_I_k1 = FFTW_ifft(np.where(in_k1,1,0)) # so this gets the thing for each k-bin
    fft_I_k2 = FFTW_ifft(np.where(in_k2,1,0)) # so this gets the thing for each k-bin

    N_bins = len(theta)
    #set our number of bins for k3 from theta
    bins = np.zeros((N_bins,2))
    for i in range(N_bins):
        bins[i,:] = [k3[i]-s*k_f,k3[i]+s*k_f]

    fft_I = np.zeros((N_bins,N_side,N_side,N_side),dtype=dtype_r)
    In_bin = np.zeros((N_bins,*k_mag.shape),dtype=np.bool_)
    for i in range(N_bins): #for each bin a new FFT box is created to get Delta(n,k) and I(n,k) after being IFFTed
        In_bin[i] = np.where(np.logical_and(k_mag>bins[i,0],k_mag<bins[i,1]),True,False)#boolean type whether each k-vector is in bin
        ifftbox1 = np.where(In_bin[i],1,0)
        fft_I[i] = FFTW_ifft(ifftbox1) # so this gets the thing for each k-bin

    #now compute Ntri
    Ntri = np.zeros(N_bins,dtype=dtype_r)
    #loops over each bin to calculate bk
    for i in range(N_bins):
        Ntri[i] = np.sum((fft_I_k1*fft_I_k2*fft_I[i])) # number of triangles in bin

    #removes divide by zero errors (bk will be zero for 0 Ntri)
    Ntri = np.where(Ntri==0,1,Ntri)
    return In_bin,Ntri,in_k1,in_k2


def bk_equalateral_compute_bins(k_eq,N_side,s,k_mag,k_f,iso_f = 1,dtype=np.complex64,threads=1,rfft=False): #computes binning information for power spectrum #this is dependent on compute gird info
        #raise warning if bad dtype
    if dtype != np.complex128 and dtype != np.complex64:
        raise Exception("Invalid dtype")

    #import which fft type we want to use for the estimator
    #also sets some values in the case of real FFTs - i.e. fourier space arrays are halved
    if rfft:
        N_side_r = N_side//2 + 1                   #N_side changes in last axis for rffts
        if dtype == np.complex128:
            from FFTW import FFTW_irfft_double as iFFT_
            dtype_r = np.float64                            #dtype changes in real space for rffts
        elif dtype == np.complex64:
            from FFTW import FFTW_irfft_single as iFFT_
            dtype_r = np.float32
    else:
        if dtype == np.complex128:
            from FFTW import FFTW_ifft_double as iFFT_
        elif dtype == np.complex64:
            from FFTW import FFTW_ifft_single as iFFT_
        dtype_r = dtype                         
        N_side_r = N_side                       

    #wrapper function for iFFT scheme imported from FFTW module - counts iFFTs
    def FFTW_ifft(delta):
        #global iFFT_number
        #iFFT_number += 1
        return iFFT_(delta,threads)

    #set our number of bins for k3 from theta
    bins=np.zeros((len(k_eq),2))
    for i in range(len(k_eq)):
        bins[i,:] = [k_eq[i]-s*k_f,k_eq[i]+s*k_f]
    
    if iso_f != 1:            # if isoceles then calculate these bins 
        k_s = (iso_f)*k_eq    
        bins1=np.zeros((len(k_s),2))
        for i in range(len(k_s)):
            bins1[i,:] = [k_s[i]-s*k_f,k_s[i]+s*k_f]

    N_bins = len(k_eq) #number of bins

    fft_I = np.zeros((N_bins,N_side,N_side,N_side),dtype=dtype_r)
    In_bin = np.zeros((N_bins,*k_mag.shape),dtype=np.bool_)
    for i in range(N_bins): #for each bin a new FFT box is created to get Delta(n,k) and I(n,k) after being IFFTed
        In_bin[i] = np.where(np.logical_and(k_mag>bins[i,0],k_mag<bins[i,1]),True,False)#boolean type whether each k-vector is in bin
        ifftbox1 = np.where(In_bin[i],1,0)
        fft_I[i] = FFTW_ifft(ifftbox1) # so this gets the thing for each k-bin
    
    if iso_f != 1:
        fft_I_iso = np.zeros((N_bins,N_side,N_side,N_side),dtype=dtype_r)
        In_bin1 = np.zeros((N_bins,*k_mag.shape),dtype=np.bool_)
        for i in range(N_bins): #for each bin a new FFT box is created to get Delta(n,k) and I(n,k) after being IFFTed
            In_bin1[i] = np.where(np.logical_and(k_mag>bins1[i,0],k_mag<bins1[i,1]),True,False)#boolean type whether each k-vector is in bin1
            ifftbox1 = np.where(In_bin[i],1,0)
            fft_I_iso[i] = FFTW_ifft(ifftbox1) # so this gets the thing for each k-bin
    
    #now compute Ntri
    Ntri = np.zeros(N_bins,dtype=dtype_r)
    if iso_f == 1:
        #loops over each bin to calculate bk
        for i in range(N_bins):
            Ntri[i] = np.sum((fft_I[i]*fft_I[i]*fft_I[i])) # number of triangles in bin
    else:
        #loops over each bin to calculate bk
        for i in range(N_bins):
            Ntri[i] = np.sum((fft_I_iso[i]*fft_I_iso[i]*fft_I[i])) # number of triangles in bin
    
    #removes divide by zero errors (bk will be zero for 0 Ntri)
    Ntri = np.where(Ntri==0,1,Ntri)
    if iso_f == 1:
        return In_bin,Ntri,In_bin #In_bin used both times
    else:
        return In_bin,Ntri,In_bin1

#separate func just for binning info -
#theta binning scheme
#find the modes in bins for k1 k2 and k3 and then computes the number of closed triangles using iFFTs.
#@jit(nopython=True)
"""
def bk_theta_compute_bins1(k1,k2,theta,N_side,s,k_mag,k_f,dtype=np.complex64,threads=1,rfft=False): #computes binning information for power spectrum #this is dependent on compute gird info
        #raise warning if bad dtype
    if dtype != np.complex128 and dtype != np.complex64:
        raise Exception("Invalid dtype")

    k3 = np.sqrt((k2*np.sin(theta))**2 + (k2*np.cos(theta)+k1)**2) #get k3

    k1bin = [k1-s*k_f,k1+s*k_f] #maybe change to s/2
    k2bin = [k2-s*k_f,k2+s*k_f]

    in_k1 = np.logical_and(k_mag>k1bin[0],k_mag<k1bin[1]) # defines k-vectors in bin
    in_k2 = np.logical_and(k_mag>k2bin[0],k_mag<k2bin[1]) # defines k-vectors in bin

    return In_bin,Ntri,in_k1,in_k2"""