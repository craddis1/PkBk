import numpy as np
#from time import time # maybe add computation time feature
from itertools import combinations_with_replacement
from numba import jit, complex64, complex128,prange
#import sys
#sys.path.append('../Library')
import base_funcs as bf


#intialize class Bk 
class Bk:
    def __init__(self,delta,l,L,N_side,ks,grid_info,binning_info,rs=[0,0],exorder=0,dtype=np.complex64,threads=1,real=False,verbose=True,fastmath=True):

#unpack variables from grid info and binning info...
        xi,x_norm,ki,k_mag,MAS,k_f,k_ny = grid_info
        In_bin,Ntri = binning_info
        
        N_bins = len(ks)
        #raise warning if bad dtype
        if dtype != np.complex128 and dtype != np.complex64:
            raise Exception("Invalid dtype")
            
        #import which fft type we want to use for the estimator
        #also sets some values in the case of real FFTs - i.e. fourier space arrays are halved
        if real:
            N_side_r = N_side//2 + 1                   #N_side changes in last axis for rffts
            if dtype == np.complex128:
                from FFTW import FFTW_irfft_double as iFFT_
                from FFTW import FFTW_rfft_double as FFT_       
                dtype_r = np.float64                            #dtype changes in real space for rffts
            elif dtype == np.complex64:
                from FFTW import FFTW_irfft_single as iFFT_
                from FFTW import FFTW_rfft_single as FFT_
                dtype_r = np.float32
        else:
            if dtype == np.complex128:
                from FFTW import FFTW_ifft_double as iFFT_
                from FFTW import FFTW_fft_double as FFT_         
            elif dtype == np.complex64:
                from FFTW import FFTW_ifft_single as iFFT_
                from FFTW import FFTW_fft_single as FFT_
            dtype_r = dtype                         
            N_side_r = N_side                       
        
        #wrapper function for FFT scheme imported from FFTW module - counts FFTs
        def FFTW_fft(delta):
            global FFT_number
            FFT_number += 1
            return FFT_(delta,threads)
        #wrapper function for iFFT scheme imported from FFTW module - counts iFFTs
        def FFTW_ifft(delta):
            global iFFT_number
            iFFT_number += 1
            return iFFT_(delta,threads)

        #calulates G_i the sum over k_i Q_i fields etc
        def Qpqrs(delta,xi,ki,l): # extends
            if l == 0:
                return FFTW_fft(delta) # this is need for the way Pk_bisector is implemented

            kQ_sum = np.zeros((N_side,N_side,N_side_r)) # create empty array

            ind = np.array(list(combinations_with_replacement(np.arange(3), l))) #find all combinations for l components from x,y,z 
            for i in ind: # so iterates for each possible ij etc

                k_prod = np.prod(ki[i],axis=0) # sum kx etc arrays for each combination
                x_prod = np.prod(xi[i],axis=0) #real parts
                #count distinct permutation for each combination
                kQ_sum = np.add(kQ_sum,np.multiply(bf.count_distinct_permutations(i)*k_prod,FFTW_fft(delta*x_prod),dtype=dtype)) # e.g. all terms in square brackets eq.20 scoccimarro

            return kQ_sum

        #so this is the the same as above but allows for n (or p) > 0
        #hence there is no sum over fields so two fields shape (((l+1)(l+2)/2),Nside,Nside,Nside)) is returned
        def Qpqrs2(delta,delta_k,xi,ki,l): # extends
            if l == 0:
                return delta_k # this is need for the way Pk_bisector is implemented
            N_side = int(len(delta[:,0,0]))
            kQ_sum = np.zeros((N_side,N_side,N_side_r)) # create empty array
            
            ind = np.array(list(combinations_with_replacement(np.arange(3), l))) #find all combinations for l components from x,y,z 
            #print(ind)
            kF = np.zeros((len(ind),N_side,N_side,N_side_r),dtype=dtype)
            Q_ij = np.zeros((len(ind),N_side,N_side,N_side_r),dtype=dtype)
            for i in range(len(ind)): # so iterates for each possible ij etc

                k_prod = np.prod(ki[ind[i]],axis=0) # sum ki kj etc arrays for each combination
                x_prod = np.prod(xi[ind[i]],axis=0) #configuration space parts e.g. xi xj
                #count distinct permutation for each combination
                kF[i] = bf.count_distinct_permutations(ind[i])*k_prod*delta_k #k1 terms
                Q_ij[i] = FFTW_fft(delta*x_prod) #delta_k2,delta_k3 convolved x terms

            return kF,Q_ij

        #this is pretty nice and beefy function used when we have powers of the dot products... #successor of Qpqrs2
        # (k1.x_{field})^l1 (x1.x2)^l2
        def Fields_func(delta,k1,xi,x_norm,l1,l2,field=1):
            #for the two cartesian expansions
            ind1 = np.array(list(combinations_with_replacement(np.arange(3), l1))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
            ind2 = np.array(list(combinations_with_replacement(np.arange(3), l2)))
            
            F_1 = np.zeros((len(ind2),N_side,N_side,N_side_r),dtype=dtype) # create empty array
            F_2 = np.zeros((len(ind2),N_side,N_side,N_side_r),dtype=dtype) # create empty array

            delta1 = delta*1  #assign fields
            delta2 = delta*1
            if field == 2:
                delta2 *= 1/(x_norm**(l1+2*l2))
            else:
                delta1 *= 1/(x_norm**(l1+2*l2))
            
            # so iterates for each possible ij etc   
            for i in range(len(ind2)):  # this for x1.x2 part
                x_prod2 = np.prod(xi[ind2[i]],axis=0)
                F_2[i] = bf.count_distinct_permutations(ind2[i])*FFTW_fft(delta2*x_prod2) # e.g. all terms in square brackets eq.20 scoccimarro
                for j in ind1: #for k1.x1 part #this is then condensed...
                    k_prod = np.prod(k1[j],axis=0) # sum kx etc arrays for each combination
                    x_prod = np.prod(xi[j],axis=0) #sum xi etc
                    F_1[i] += bf.count_distinct_permutations(j)*k_prod*FFTW_fft(delta1*x_prod*x_prod2) #

            return F_1,F_2
        
        # (k1.x_i)^l1 (x1.xi)^l2 #so new stuff needed for bispectrum - more like Qpqrs2
        # will returns probably 3,3
        #returns 2 arrays shape (Num_terms(l2),Num_terms(l1),N_side,N_side,N_side)
        def Fields_funcbk(delta,k1,xi,x_norm,l1,l2):
            #for the two cartesian expansions
            ind1 = np.array(list(combinations_with_replacement(np.arange(3), l1))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
            ind2 = np.array(list(combinations_with_replacement(np.arange(3), l2))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
            
            #now all 3 fields the same length!
            F_1 = np.zeros((len(ind1),len(ind2),N_side,N_side,N_side_r),dtype=dtype) # create empty array
            F_2 = np.zeros((len(ind1),len(ind2),N_side,N_side,N_side_r),dtype=dtype) # create empty array

            delta1 = delta/(x_norm**(l1+2*l2))  #assign fields
            
            # so iterates for each possible ij etc   
            for i in range(len(ind2)):  # this for x1.x2 part
                x_prod2 = np.prod(xi[ind2[i]],axis=0)
                
                for j in range(len(ind1)): #for k1.xi part -no longer condensed
                    k_prod = np.prod(k1[ind1[j]],axis=0) # sum kx etc arrays for each combination
                    F_1[i,j] = bf.count_distinct_permutations(ind1[j])*k_prod*FFTW_fft(delta1*x_prod2) 
                    #x_prod = np.prod(xi[ind1[j]],axis=0) #sum xi etc
                    #F_2[i,j] = bf.count_distinct_permutations(ind2[i])*FFTW_fft(delta*x_prod*x_prod2) 
                            
            return F_1,F_2
                            
        # (k1.x_1)^l1 (x1.x2)^l2 (x1.x3)^l2 
        #(k.x1)(x2.x1)(x3.x1)
        #mmm only really new bit is the F_1 bit being 
        def Fields_funcbk3(delta,ki,xi,x_norm,l1,l2):
            #for the two cartesian expansions
            ind1 = np.array(list(combinations_with_replacement(np.arange(3), l1))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
            ind2 = np.array(list(combinations_with_replacement(np.arange(3), l2))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
            
            #now all 3 fields the same length!
            F_1 = np.zeros((len(ind2),len(ind2),N_side,N_side,N_side_r),dtype=dtype) # create empty array

            delta1 = delta/(x_norm**(l1+4*l2))  #assign fields
            
            # so iterates for each possible ij etc   
            for i in range(len(ind2)):  # this for x1.xi part
                x_prod2 = np.prod(xi[ind2[i]],axis=0)
                for j in range(len(ind2)): #for k1.xj part 
                    x_prod3 = np.prod(xi[ind2[i]],axis=0)
                    for k in range(len(ind1)): #for k1.xi1 part - condensed

                        k_prod = np.prod(ki[ind1[k]],axis=0) # sum kx etc arrays for each combination
                        x_prod = np.prod(xi[ind1[k]],axis=0) #sum xi etc
                        F_1[i,j] += bf.count_distinct_permutations(ind1[k])*k_prod*FFTW_fft(delta1*x_prod*x_prod2*x_prod3)
             
            return F_1

        #Lengendre multipoles
        def delta_lm(delta_x,delta_k,xi,ki,l):  #convolves delta with legendre polynomial
            if l == 0:
                return delta_k
            if l==1:
                Q_x = Qpqrs(delta_x/x_norm**l,xi,ki,l)
                return Q_x
            if l==2:
                Q_xx = Qpqrs(delta_x/x_norm**l,xi,ki,l)
                return (3/2)*Q_xx - (1/2)*delta_k
            if l==3:
                Q_xx = Qpqrs(delta_x/x_norm**l,xi,ki,l)
                return (1/2)*(5*Q_xx - 3*delta_lm(delta_x,delta_k,xi,ki,1))
            if l==4:
                Q_xxxx = Qpqrs(delta_x/x_norm**l,xi,ki,l)
                return (35/8)*Q_xxxx - (5/2)*delta_lm(delta_x,delta_k,xi,ki,2) - (7/8)*delta_k
            else:
                raise Exception(l,"l-multipole not implemented")

        # bispecrum with theta binning and legendre polynomials decomposition #with endpoint
        def Bk_endpoint_full(delta,l):
            global FFT_number,iFFT_number
            FFT_number = 0
            iFFT_number = 0

            if verbose: print('nyquist frequency=',k_ny,'fundamental frequency=',k_f)
            delta_k = FFTW_fft(delta)
            
            #constants - e.g. see watkinson
            Npix = N_side**3
            V = L**3
            H = V/Npix
            const = H**3 *(1/V)
            
            ifft_del = ifft_field(delta_k)
            if l != 0:
                #calulate delta_lm - this needs to be done once for each harmonic
                F_lm = delta_lm(delta,delta_k,xi,ki,l)
                ifft_del1 = ifft_field(F_lm)
            else:
                ifft_del1 = ifft_del

            #compute sum 
            Bbin_lm = ifft_sum(ifft_del1,ifft_del,ifft_del)
            if verbose: print('Number of FFTs used=',FFT_number,'Number of iFFTs used=',iFFT_number)
            return (2*l+1)*const*Bbin_lm/Ntri
        
        def ifft_field(field,single_field=True):#lets edit this...
            if single_field:
                ifft_F = np.zeros((N_bins,N_side**3),dtype=dtype_r) #create empty arrays for the iffts of each bins
                for i in range(N_bins):#for each bin a new FFT box is created to get I(n,k) after being IFFTed
                    ifft_F[i] = FFTW_ifft(np.where(In_bin[i],field*MAS,0)).flatten()
            else:
                ifft_F = np.zeros((N_bins,field.shape[0],N_side**3),dtype=dtype_r) #create empty arrays for the iffts of each bins
                for i in range(N_bins):#for each bin a new FFT box is created to get I(n,k) after being IFFTed
                    for j in range(field.shape[0]):
                        ifft_F[i,j] = FFTW_ifft(np.where(In_bin[i],field[j]*MAS,0)).flatten()
            return ifft_F
        
        @jit(nopython=True,fastmath=fastmath,parallel=True)#
        def ifft_sum(field1: complex64[:, :], field2: complex64[:, :], field3: complex64[:, :]) -> complex64[:, :, :]: # does sum over fields where there are possible closed triangles...      
            Bk_lm = np.zeros((N_bins,N_bins,N_bins),dtype=np.complex64)
            for i in prange(N_bins):
                for j in range(N_bins):
                    for k in range(N_bins):
                        if (i > j+k+2) or ((j > i+k+2)or (k > i+j+2)):#these values should all be 0  as dirac delta
                            continue
                        sum_real = 0.0
                        sum_imag = 0.0
                        for idx in range(field1.shape[1]):
                            sum_real += field1.real[i, idx] * field2.real[j, idx] * field3.real[k, idx] - field1.imag[i, idx] * field2.imag[j, idx] * field3.real[k, idx] - field1.real[i, idx] * field2.imag[j, idx] * field3.imag[k, idx] - field1.imag[i, idx] * field2.real[j, idx] * field3.imag[k, idx]
                            sum_imag += field1.real[i, idx] * field2.imag[j, idx] * field3.real[k, idx] + field1.imag[i, idx] * field2.real[j, idx] * field3.real[k, idx] + field1.real[i, idx] * field2.real[j, idx] * field3.imag[k, idx] - field1.imag[i, idx] * field2.imag[j, idx] * field3.imag[k, idx]
                        Bk_lm[i, j, k] = sum_real + 1j*sum_imag

            return Bk_lm
        
        @jit(nopython=True,fastmath=fastmath,parallel=True)#
        def ifft_sum1(field1,field2,field3): # does sum over fields where there are possible closed triangles...      
            Bk_lm = np.zeros((N_bins,N_bins,N_bins),dtype=np.complex64)
            for i in prange(N_bins):
                for j in range(N_bins):# i+1 -> k1 >= k2
                    for k in range(N_bins):#j+1 -> k2 >= k3
                        if (i > j+k+2) or ((j > i+k+2)or (k > i+j+2)):#these values should all be 0 as dirac delta
                            continue
                        sum_real = np.float32(0.0)
                        sum_imag = np.float32(0.0)
                        for m in range(field1.shape[1]):
                            for idx in range(field1.shape[-1]):
                                sum_real = sum_real + field1.real[i,m, idx] * field2.real[j,m, idx] * field3.real[k, idx] - field1.imag[i,m, idx] * field2.imag[j,m, idx] * field3.real[k, idx] - field1.real[i,m, idx] * field2.imag[j,m, idx] * field3.imag[k, idx] - field1.imag[i,m, idx] * field2.real[j,m, idx] * field3.imag[k, idx]
                                sum_imag = sum_imag + field1.real[i,m, idx] * field2.imag[j,m, idx] * field3.real[k, idx] + field1.imag[i,m, idx] * field2.real[j,m, idx] * field3.real[k, idx] + field1.real[i,m, idx] * field2.real[j,m, idx] * field3.imag[k, idx] - field1.imag[i,m, idx] * field2.imag[j,m, idx] * field3.imag[k, idx]
                        Bk_lm[i, j, k] = sum_real + 1j*sum_imag
            return Bk_lm

        @jit(nopython=True,parallel=True,fastmath=fastmath)
        def ifft_sum2(field1,field2,field3): # does sum over fields where there are possible closed triangles...      
            Bk_lm = np.zeros((N_bins,N_bins,N_bins),dtype=np.complex64)
            for i in prange(N_bins):
                for j in range(N_bins):# i+1 -> k1 >= k2
                    for k in range(N_bins):#j+1 -> k2 >= k3
                        if (i > j+k+2) or ((j > i+k+2)or (k > i+j+2)):#these values should all be 0 as dirac delta
                            continue
                        sum_real = np.float32(0.0)
                        sum_imag = np.float32(0.0)
                        for m in range(field1.shape[1]):
                            for idx in range(field1.shape[-1]):
                                sum_real = sum_real + field1.real[i,m, idx] * field2.real[j, idx] * field3.real[k,m, idx] - field1.imag[i,m, idx] * field2.imag[j, idx] * field3.real[k,m, idx] - field1.real[i,m, idx] * field2.imag[j, idx] * field3.imag[k,m, idx] - field1.imag[i,m, idx] * field2.real[j, idx] * field3.imag[k,m, idx]
                                sum_imag = sum_imag + field1.real[i,m, idx] * field2.imag[j, idx] * field3.real[k,m, idx] + field1.imag[i,m, idx] * field2.real[j, idx] * field3.real[k,m, idx] + field1.real[i,m, idx] * field2.real[j, idx] * field3.imag[k,m, idx] - field1.imag[i,m, idx] * field2.imag[j, idx] * field3.imag[k,m, idx]
                        Bk_lm[i, j, k] = sum_real + 1j*sum_imag
            return Bk_lm
        
        @jit(nopython=True,parallel=True,fastmath=fastmath)
        def ifft_mixed_sum(field1,field2,field3): # does sum over fields where there are possible closed triangles...
            Bk_lm = np.zeros((N_bins,N_bins,N_bins),dtype=dtype_r)
            for i in range(N_bins):
                for j in range(N_bins):
                    for k in range(N_bins):
                        if (i > j+k+2) or ((j > i+k+2)or (k > i+j+2)):#these values should all be 0 as dirac delta
                            continue
                        sum_real = np.float32(0.0)
                        sum_imag = np.float32(0.0)
                        for m in range(field1.shape[1]):
                            for n in range(field1.shape[2]):
                                for idx in range(field1.shape[-1]):
                                    sum_real = sum_real + field1.real[i,m,n, idx] * field2.real[j,m, idx] * field3.real[k,n, idx] - field1.imag[i,m,n, idx] * field2.imag[j,m, idx] * field3.real[k,n, idx] - field1.real[i,m,n, idx] * field2.imag[j,m, idx] * field3.imag[k,n, idx] - field1.imag[i,m,n, idx] * field2.real[j,m, idx] * field3.imag[k,n, idx]
                                    sum_imag = sum_imag + field1.real[i,m, idx] * field2.imag[j,m, idx] * field3.real[k,n, idx] + field1.imag[i,m,n, idx] * field2.real[j,m, idx] * field3.real[k,n, idx] + field1.real[i,m,n, idx] * field2.real[j,m, idx] * field3.imag[k,n, idx] - field1.imag[i,m,n, idx] * field2.imag[j,m, idx] * field3.imag[k,n, idx]
                        Bk_lm[i, j, k] = sum_real + 1j*sum_imag
            return Bk_lm
        
        @jit(nopython=True,parallel=True,fastmath=fastmath)
        def ifft_mixed_sum1(field1,field2,field3): # does sum over fields where there are possible closed triangles...
            Bk_lm = np.zeros((N_bins,N_bins,N_bins),dtype=dtype_r)
            for i in range(N_bins):
                for j in range(N_bins):
                    for k in range(N_bins):
                        if (i > j+k+2) or ((j > i+k+2)or (k > i+j+2)):#these values should all be 0 as dirac delta
                            continue
                        sum_real = np.float32(0.0)
                        sum_imag = np.float32(0.0)
                        for m in range(field1.shape[1]):
                            for n in range(field1.shape[2]):
                                for idx in range(field1.shape[-1]):
                                    sum_real = sum_real + field1.real[i,m,n, idx] * field2.real[j,n, idx] * field3.real[k,m, idx] - field1.imag[i,m,n, idx] * field2.imag[j,n, idx] * field3.real[k,m, idx] - field1.real[i,m,n, idx] * field2.imag[j,n, idx] * field3.imag[k,m, idx] - field1.imag[i,m,n, idx] * field2.real[j,n, idx] * field3.imag[k,m, idx]
                                    sum_imag = sum_imag + field1.real[i,m, idx] * field2.imag[j,n, idx] * field3.real[k,m, idx] + field1.imag[i,m,n, idx] * field2.real[j,n, idx] * field3.real[k,m, idx] + field1.real[i,m,n, idx] * field2.real[j,n, idx] * field3.imag[k,m, idx] - field1.imag[i,m,n, idx] * field2.imag[j,n, idx] * field3.imag[k,m, idx]
                        Bk_lm[i, j, k] = sum_real + 1j*sum_imag
            return Bk_lm  
        
        @jit(nopython=True,parallel=True,fastmath=fastmath)
        def ifft_sumx2x3(field1,field2,field3): # does sum over fields where there are possible closed triangles...      
            Bk_lm = np.zeros((N_bins,N_bins,N_bins),dtype=np.complex64)
            for i in prange(N_bins):
                for j in range(N_bins):# i+1 -> k1 >= k2
                    for k in range(N_bins):#j+1 -> k2 >= k3    
                        if (i > j+k+2) or ((j > i+k+2)or (k > i+j+2)):#these values should all be 0 as dirac delta
                            continue
                        sum_real = np.float32(0.0)
                        sum_imag = np.float32(0.0)
                        for m in range(field2.shape[1]):
                            for idx in range(field1.shape[-1]):
                                sum_real = sum_real + field1.real[i, idx] * field2.real[j,m, idx] * field3.real[k,m, idx] - field1.imag[i, idx] * field2.imag[j,m, idx] * field3.real[k,m, idx] - field1.real[i, idx] * field2.imag[j,m, idx] * field3.imag[k,m, idx] - field1.imag[i, idx] * field2.real[j,m, idx] * field3.imag[k,m, idx]
                                sum_imag = sum_imag + field1.real[i, idx] * field2.imag[j,m, idx] * field3.real[k,m, idx] + field1.imag[i, idx] * field2.real[j,m, idx] * field3.real[k,m, idx] + field1.real[i, idx] * field2.real[j,m, idx] * field3.imag[k,m, idx] - field1.imag[i, idx] * field2.imag[j,m, idx] * field3.imag[k,m, idx]
                        Bk_lm[i, j, k] = sum_real + 1j*sum_imag
            return Bk_lm
        
        #non-endpoint LOS - all multipoles
        # bispecrum with theta binning and legendre polynomials decomposition
        def Bk_full(delta,l):
            if verbose: print('nyquist frequency=',k_ny,'fundamental frequency=',k_f)
            global FFT_number,iFFT_number
            FFT_number = 0
            iFFT_number = 0
            
            delta_k = FFTW_fft(delta)
            
            #constants - e.g. see watkinson 
            Npix = N_side**3; V = L**3
            H = V/Npix
            const = H**3 *(1/V)
            
            #main computation...
            def main_func(l):
                if l == 0:
                    return self.mono

                if l==1:
                    
                    #set up the first fields
                    kQ1 = Qpqrs(delta/x_norm,xi,ki,1) 
                    ifft_G1_1 = ifft_field(kQ1) # ifft
                    ifft_F = ifft_field(delta_k)
                    
                    # for: k1.x1
                    Bk_lm = ifft_sum(ifft_G1_1,ifft_F,ifft_F)
                    
                    # ok so now consider if we outside d = x1
                    if exorder == 'mid': # so here we compute the other endpoints as it were...
                        #using the x1 + x2 + x3 = 3d
                        
                        Bk_lm1 = np.zeros((3,N_bins,N_bins,N_bins),dtype=dtype_r) # I want 3 results from this as the other 2 endpoints are useful to have...
                        Bk_lm1[0] = Bk_lm
                        
                        kF_k1,Q_ij_x = Qpqrs2(delta/x_norm,delta_k,xi,ki,1)
                        
                        ifft_kF = ifft_field(kF_k1,False)
                        ifft_Q1_x = ifft_field(Q_ij_x,False)

                        #k.x2    
                        Bk_lm1[1] = ifft_sum1(ifft_kF,ifft_Q1_x,ifft_F)
                        
                        #k.x3   
                        Bk_lm1[2] = ifft_sum2(ifft_kF,ifft_F,ifft_Q1_x)
                        
                        Bk_lm = Bk_lm1
                           
                    if exorder == 1:
                        r,s = rs #unpack r and s 
                        #this stuff is used in both r and s parts...
                        #for the - k. xi parts
                        kF_k, Q_ij = Qpqrs2(delta,FFTW_fft(delta/x_norm),xi,ki,1) #get field in k-space
                        ifft_kF_x = ifft_field(kF_k,False)
                        ifft_Q1 = ifft_field(Q_ij,False)
                        
                        #for the - x1 (x1.xi) parts
                        F_1,F_2  = Fields_func(delta,ki,xi,x_norm,1,1)
                        ifft_G1Q1_x3 = ifft_field(F_1,False)

                        if r > 0:
                            #for r k.x2
                            Bk_lm += r*ifft_sum1(ifft_kF_x,ifft_Q1,ifft_F)
                        
                            #for: - r x1 (x1.x2)   #similar to stuff above but with Fields func rather than Qpqrs
                            Bk_lm += -r*ifft_sum1(ifft_G1Q1_x3,ifft_Q1,ifft_F)
                        
                        if s > 0:
                            #for s k.x3              
                            #uses kF_k1 from before and Q_ij
                            Bk_lm += s*ifft_sum2(ifft_kF_x,ifft_F,ifft_Q1)

                            #for -s x1 (x1.x3) #similar to before but binning is now with k3
                            Bk_lm += -s*ifft_sum2(ifft_G1Q1_x3,ifft_F,ifft_Q1)
                            
                    
                    if exorder == 2:#should be careful about bf.count... as we reuse stuff but should be ok as little is of second order...-but CHECK!!!
                        r,s = rs #unpack r and s 
                        #this is going to be fucking long but here we go hope it is interesting!
                        #can't split r and s now...
                        
                        kF_k, Q_ij = Qpqrs2(delta,FFTW_fft(delta/x_norm),xi,ki,1) #get field in k-space
                        ifft_kF_x = ifft_field(kF_k,False)
                        ifft_Q1 = ifft_field(Q_ij,False)
                        
                        Bk_lm1 = np.zeros((15,N_bins,N_bins,N_bins),dtype=dtype_r)
                        Bk_lm1[0] = Bk_lm
                        
                        #for r(1+r+s) k.x2
                        Bk_lm1[1] = r*(1+r+s)*ifft_sum1(ifft_kF_x,ifft_Q1,ifft_F)
                        
                        #for s(1+r+s) k.x3           
                        Bk_lm1[2] = s*(1+r+s)*ifft_sum2(ifft_kF_x,ifft_F,ifft_Q1)
                         
                        #for the - x1 (x1.xi) parts-------------------------------------------------
                        F_1,F_2  = Fields_func(delta,ki,xi,x_norm,1,1)
                        ifft_G1Q1_x3 = ifft_field(F_1,False)
                        
                        #for  -r*(1+r+s) x1 (x1.x2) parts
                        Bk_lm1[3] = -r*(1+r+s)*ifft_sum1(ifft_G1Q1_x3,ifft_Q1,ifft_F)         
                        
                        #for  -s*(1+r+s)*x1 (x1.x3) #similar to before but binning is now with k3
                        Bk_lm1[4] = -s*(1+r+s)*ifft_sum2(ifft_G1Q1_x3,ifft_F,ifft_Q1)
                               
                        #ok so now for x1 (xi.x1)^2    v.similar to x1 (x1.x2) parts--------------- 
                        F_1,F_2  = Fields_func(delta,ki,xi,x_norm,1,2)
                        ifft_G1Q2_x5 = ifft_field(F_1,False)
                        ifft_Q2 = ifft_field(F_2,False)
                        
                        #for (3/2)* r**2 x1 (x2.x1)^2
                        Bk_lm1[5] += (3/2)* r**2*ifft_sum1(ifft_G1Q2_x5,ifft_Q2,ifft_F)          
                        
                        #for (3/2)* s**2 x1 (x1.x3)^2 #similar to before but binning is now with k3
                        Bk_lm1[6] += (3/2)* s**2*ifft_sum2(ifft_G1Q2_x5,ifft_F,ifft_Q2)
                        
                        #ok now for xi (xi.x1)------------------------------------------------------
                        F_1,F_2  = Fields_funcbk(delta,ki,xi,x_norm,1,1) # returns a (3,3,grid.shape)
                        ifft_Q1k_x3 = ifft_field(F_1.reshape((F_1.shape[0]*F_1.shape[1],N_side**3)),False)#reshape into (9,grid.shape)  
                                                
                        #for -r^2 x2 (x1.x2)
                        Bk_lm1[7] = -r**2 *ifft_sum1(ifft_Q1k_x3,bf.reshape69(ifft_Q2),ifft_F)
                        
                        #for -s^2 x3 (x1.x3)  
                        Bk_lm1[8] = -s**2 *ifft_sum2(ifft_Q1k_x3,ifft_F,bf.reshape69(ifft_Q2))
                        
                        #ok now for xi (xj.x1) parts----------------------------------------------
                        #for -r s x3(x1.x2)
                        Bk_lm1[9] = -r*s *ifft_mixed_sum1(ifft_Q1k_x3.reshape((N_bins,3,3,N_side**3)),ifft_Q1,ifft_Q1)
                        #for -r s x2(x1.x3)# same as above but kQ1 pairings are swapped!
                        Bk_lm1[10] = -r*s *ifft_mixed_sum(ifft_Q1k_x3.reshape((N_bins,3,3,N_side**3)),ifft_Q1,ifft_Q1)
                         
                        #ok now for the x1 (xi.xi)-------------------------------------------------- 
                        F_1 = Qpqrs(delta/x_norm**3,xi,ki,1)
                        ifft_G_x3 = ifft_field(F_1)
                        F_2 = FFTW_fft(delta*x_norm**2)
                        ifft_xx = ifft_field(F_2)
                        
                         #for -(1/2) r^2 x1 (x2.x2)
                        Bk_lm1[11] = -(1/2)*r**2 *ifft_sum(ifft_G_x3,ifft_xx,ifft_F)
                        
                        #for -(1/2) s^2 x1 (x3.x3)
                        Bk_lm1[12] = -(1/2)*s**2 *ifft_sum(ifft_G_x3,ifft_F,ifft_xx)
                        
                        #ok now for -rs x1 (x2.x3)-------------------------------------------------------
                        Bk_lm1[13] = - r*s*ifft_sum_x2x3(ifft_G_x3,ifft_Q1,ifft_Q1)
                            
                        #ok now for the final large boi---------------------------------------------------
                        #3rs (k.x1)(x2.x1)(x3.x1)
                        #ok so now we can reuse field1 from the x1 (x2.x1)^2 part but need to reshape...
                        #is true as permutations were added to the non x1 part
                        ifft_GQQ_x5 = bf.reshape69(ifft_G1Q2_x5,True).reshape((N_bins,3,3,N_side**3))
                        
                        Bk_lm1[14] = 3*r*s *ifft_mixed_sum(ifft_GQQ_x5,ifft_Q1,ifft_Q1)
                            
                return  const/Ntri*Bk_lm


            def Bk_legendre(l):
                if l == 0:
                    return self.mono#main_func(0)
                if l == 1:
                    return main_func(1)
                if l == 2:
                    return (1/2)*(3*main_func(2)-self.mono)
                if l == 3:
                    return (1/2)*(5*main_func(3)- 3*main_func(1))
                if l == 4:
                    return (1/8)*(35*main_func(4)- 30*main_func(2) + 3*self.mono)
                else:
                    raise "Multipole not implemented"

            Bbin_lm = Bk_legendre(l)
            if verbose: print('Number of FFTs used=',FFT_number,'Number of iFFTs used=',iFFT_number)
            return (2*l+1)*Bbin_lm
        
        #if even then calculate monopole    
        if l % 2 == 0:
            # just calculate the monopole - it useful
            self.mono =  Bk_endpoint_full(delta,0)
            if l != 0:
                self.bk = Bk_full(delta,l) 
        
        else:
            # this is to pick which estimator function to call! - based on LOS and l
            #self.mono = 0
            self.bk = Bk_full(delta,l)