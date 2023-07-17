# so the has theta binning endpoint approximation

""" Up to date 16/3/2023 - neatened main func maybe add more optimal dipole and quad?
Functions to call: These deal with all multipoles

General: Bk_theta_iFFT.Bk(delta,l,L,N_side,theta,grid_info,binning_info,rs [0,0],dtype=np.complex64,threads=1,real=False,verbose=True)

Either calls:
Bk_endpoint_theta(delta,l) # slightly more efficient for endpoint configs - also used for monopole
Bk_theta(delta,l)

---------------------------------------------------------------------------------------------------ignore
# these are midpoint only but are slightly more effcient than the general midpoint counterpart due to less redundancy in computing terms
Bk_mono, Bk_quad = Bispectrum_FFT_theta_quad(field,L,k1,k2,theta,order,s,z_off,x_off,y_off,verbose=true) 3 less FFTs - not computing H(k) twice and 3 less iFFTs - from not iFFTing J_k twice
Bk_mono, Bk_quad = Bispectrum_FFT_theta_dipo(field,L,k1,k2,theta,order,s,z_off,x_off,y_off,verbose=true)

The optimisation is reduced for non midpoint configurations.
---------------------------------------------------------------------------------------------------ignore

Also there is class
Amnp(f_123,delta,l,L,N_side,theta,grid_info,binning_info,rs=[0,0],dtype=np.complex64,threads=1,real=False,verbose=True)
with f_123 = [m,n,p]
"""
import numpy as np
#from time import time # maybe add computation time feature
from itertools import combinations_with_replacement

import sys
sys.path.append('/home/addis/Library')
import base_funcs as bf


#intialize class Bk 
class Bk:
    def __init__(self,delta,l,L,N_side,theta,grid_info,binning_info,rs=[0,0],exorder=0,dtype=np.complex64,threads=1,real=False,verbose=True):


        #unpack variables from grid info and binning info...
        xi,x_norm,ki,k_mag,MAS,k_f,k_ny = grid_info
        In_bin,Ntri,in_k1,in_k2 = binning_info
        
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
        
        #Lengendre multipoles
        def delta_lm(delta_x,delta_k,xi,ki,l):  #convolves delta with legendre polynomial
            if l == 0:
                return delta_k
            if l==1:
                Q_x = Qpqrs(delta_x,xi,ki,l)
                return Q_x
            if l==2:
                Q_xx = Qpqrs(delta_x,xi,ki,l)
                return (3/2)*Q_xx - (1/2)*delta_k
            if l==3:
                Q_xx = Qpqrs(delta_x,xi,ki,l)
                return (1/2)*(5*Q_xx - 3*delta_lm(delta_x,delta_k,xi,ki,1))
            if l==4:
                Q_xxxx = Qpqrs(delta_x,xi,ki,l)
                return (35/8)*Q_xxxx - (5/2)*delta_lm(delta_x,delta_k,xi,ki,2) - (7/8)*delta_k
            else:
                raise Exception(l,"l-multipole not implemented")


        # bispecrum with theta binning and legendre polynomials decomposition #with endpoint
        def Bk_endpoint_theta(delta,l):
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

            delta_kk = delta_k*MAS  #get MAS_corrected field

            #calulate delta_lm - this needs to be done once for each harmonic
            F_lm = delta_lm(delta,delta_k,xi/x_norm,ki,l)*MAS 

            fft_del_k1 = FFTW_ifft(np.where(in_k1,F_lm,0)) # ifft
            fft_del_k2 = FFTW_ifft(np.where(in_k2,delta_kk,0)) # ifft
            
            N_bins = len(theta)
            fft_del = np.zeros((N_bins,N_side,N_side,N_side),dtype=dtype_r) #create empty arrays for the iffts of each bins
            for i in range(N_bins): #for each bin a new FFT box is created to get Delta(n,k) and I(n,k) after being IFFTed
                ifftbox = np.where(In_bin[i],delta_kk,0)
                fft_del[i] = FFTW_ifft(ifftbox)

            #@jit(nopython=True)
            def discrete_sum(x):
                Bbin_lm = np.zeros(x,dtype=dtype_r)
                #loops over each bin to calculate bk
                for i in range(x):
                    Un_Bbin = np.sum((fft_del_k1*fft_del_k2*fft_del[i])) # numerator of watkinson estimator
                    Bbin_lm[i] =  const * Un_Bbin/Ntri[i]
                return Bbin_lm
            
            Bbin_lm = discrete_sum(N_bins)
            if verbose: print('Number of FFTs used=',FFT_number,'Number of iFFTs used=',iFFT_number)
            return (2*l+1)*Bbin_lm


        #for Amn(2) and Amn(3) sums when n > 0 or p > 0
        #does sums for G_k as well with shape (1,Nside,Nside,Nside)
        #@jit(nopython=True)
        #loops here are the most efficeint as we have large array not summed over too many things - numba is far slower!
        def Amn_sum(N_bins,f1,f2,f3,field2=True):
            Bbin = np.zeros(N_bins,dtype=dtype_r)
            if field2:
                #loops over each bin (and composite fields!) to calculate bk
                for i in range(N_bins):
                    for j in range(len(f1)):
                        Bbin[i] += np.sum((f1[j]*f2[j]*f3[i])) # sum over 3 now real space fields!
                        
            else:
                #loops over each bin (and composite fields!) to calculate bk
                for i in range(N_bins):
                    for j in range(len(f1)):
                        Bbin[i] += np.sum((f1[j]*f2*f3[i,j])) 
                        
            return Bbin
        
        #loops here are the most efficeint as we have large array not summed over too many things - numba is far slower!
        def Amn_sum_x2x3(N_bins,f1,f2,f3): # when no x1 involved in the sum!
            Bbin = np.zeros(N_bins,dtype=dtype_r)
            #loops over each bin (and composite fields!) to calculate bk
            for i in range(N_bins):
                for j in range(len(f2)):
                    Bbin[i] += np.sum((f1*f2[j]*f3[i,j])) # sum over 3 now real space fields!
                        
            return Bbin

        #@jit(nopython=True)  #just like Amn_sum but for n > 0 AND p > 0
        def Mixed_sum(N_bins,f1,f2,f3,ordering=True):
            #f1 shape (Num_terms(f2),Num_terms(f3),Nside,Nside,Nside)
            #f2 shape (Num_terms(f2),Nside,Nside,Nside)
            #f3 shape (N_bins,Num_terms(f3),Nside,Nside,Nside)

            Bbin = np.zeros(N_bins,dtype=dtype_r)
            #Nested loop over bins and then both the 2nd and 3rd fields for bk!
            if ordering:
                for i in range(N_bins):
                    for j in range(f2.shape[0]):
                        for k in range(f3.shape[1]):
                            Bbin[i] += np.sum((f1[j,k]*f2[j]*f3[i,k])) # sum over 3 now real space fields!
            else:
                for i in range(N_bins):
                    for j in range(f2.shape[0]):
                        for k in range(f3.shape[1]):
                            Bbin[i] += np.sum((f1[j,k]*f2[k]*f3[i,j])) # sum over 3 now real space fields!
                        
            return Bbin
        
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
        
        def reshape69(arr,nobins=True):#this reshapes (xi.f)^2 function to (xi.f)(xi.g) saving redundant compuation
            if nobins:
                arr_shape = arr.shape[1:]#get underlying shape
                arr_new = np.zeros((9,*arr_shape),dtype=dtype_r)
                #now get stuff
                #[0,0]
                arr_new[0]= arr[0]
                #[0,1]
                arr_new[1] = (1/2)*arr[1]
                arr_new[3] = (1/2)*arr[1]
                #[0,2]
                arr_new[2] = (1/2)*arr[2]
                arr_new[6] = (1/2)*arr[2]
                #[1,1]
                arr_new[4] = arr[3]
                #[1,2]
                arr_new[5] = (1/2)*arr[4]
                arr_new[7] = (1/2)*arr[4]
                #[2,2]
                arr_new[8] = arr[5]
            else:
                arr_shape = arr.shape[2:]#get underlying shape
                arr_new = np.zeros((arr.shape[0],9,*arr_shape),dtype=dtype_r)

                #now get stuff
                #[0,0]
                arr_new[:,0]= arr[:,0]
                #[0,1]
                arr_new[:,1] = (1/2)*arr[:,1]
                arr_new[:,3] = (1/2)*arr[:,1]
                #[0,2]
                arr_new[:,2] = (1/2)*arr[:,2]
                arr_new[:,6] = (1/2)*arr[:,2]
                #[1,1]
                arr_new[:,4] = arr[:,3]
                #[1,2]
                arr_new[:,5] = (1/2)*arr[:,4]
                arr_new[:,7] = (1/2)*arr[:,4]
                #[2,2]
                arr_new[:,8] = arr[:,5]
            return arr_new
        
        
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

        #non-endpoint LOS - all multipoles
        # bispecrum with theta binning and legendre polynomials decomposition
        def Bk_theta(delta,l):
            if verbose: print('nyquist frequency=',k_ny,'fundamental frequency=',k_f)
            global FFT_number,iFFT_number
            FFT_number = 0
            iFFT_number = 0
            
            delta_k = FFTW_fft(delta)
            
            #constants - e.g. see watkinson 
            Npix = N_side**3
            V = L**3
            H = V/Npix
            const = H**3 *(1/V)
            
            delta_kk = delta_k*MAS  # deconvolve with MAS

            #iFFTs for Dirac_delta - get final fields
            ifft_F_k2 = FFTW_ifft(np.where(in_k2,delta_kk,0))

            N_bins = len(theta) #number of bins
            in_k3 = In_bin
            
            ifft_F_k3 = np.zeros((N_bins,N_side,N_side,N_side),dtype=dtype_r) #create empty arrays for the iffts of each bins
            for i in range(N_bins): #for each bin a new FFT box is created to get Delta(n,k) and I(n,k) after being IFFTed
                ifftbox = np.where(In_bin[i],delta_kk,0)
                ifft_F_k3[i] = FFTW_ifft(ifftbox) #delta_k3 for each k-bin
                
            Npix = N_side**3; V = L**3
            H = V/Npix
            const = H**3 *(1/V)
            #main computation...
            def main_func(l):
                if l == 0:
                    return bkmono
                
                Bk_lm = np.zeros(N_bins,dtype=dtype_r)
                if l==1:
                    # for: k1.x1
                    kQ1 = Qpqrs(delta/x_norm,xi,ki,1) 
                    ifft_G1_1 = FFTW_ifft(np.where(in_k1,kQ1*MAS,0)) # ifft
                    
                    Bk_lm = Amn_sum(N_bins,ifft_G1_1[np.newaxis,...], ifft_F_k2[np.newaxis,...],ifft_F_k3)
                    r,s = rs #unpack r and s
                    
                    if exorder == 'mid':
                        #using the x1 + x2 + x3 = 3d
                        Bk_lm1 = np.zeros((3,N_bins),dtype=dtype_r) # I want 3 results from this as the other 2 endpoints are useful to have...
                        Bk_lm1[0] = Bk_lm
                        #k.x2
                        kF_k1,Q_k2 = Qpqrs2(delta/x_norm,delta_k,xi,ki,1)
   
                        ifft_kF_k1 = np.zeros((len(Q_k2),N_side,N_side,N_side),dtype=dtype_r)
                        ifft_F_Qk2 = np.zeros((len(Q_k2),N_side,N_side,N_side),dtype=dtype_r)
                        for j in range(len(Q_k2)):
                            ifft_kF_k1[j] = FFTW_ifft(np.where(in_k1,kF_k1[j]*MAS,0))
                            ifft_F_Qk2[j] = FFTW_ifft(np.where(in_k2,Q_k2[j]*MAS,0))
                            
                        Bk_lm1[1] = Amn_sum(N_bins,ifft_kF_k1, ifft_F_Qk2,ifft_F_k3)
                        
                        #k.x3
                        ifft_F_Qk3 = np.zeros((N_bins,len(Q_k2),N_side,N_side,N_side),dtype=dtype_r)
                        for j in range(len(Q_k2)):
                            for i in range(N_bins):
                                ifft_F_Qk3[i,j] = FFTW_ifft(np.where(in_k3[i],Q_k2[j]*MAS,0))
                                
                        Bk_lm1[2] = Amn_sum(N_bins,ifft_kF_k1, ifft_F_k2,ifft_F_Qk3,False)
                        Bk_lm = Bk_lm1
                        
                    if exorder == 1:
                        
                        #this stuff is used in both r and s parts...
                        #fpr the - k. xi parts
                        kF_k, Q_ij = Qpqrs2(delta,FFTW_fft(delta/x_norm),xi,ki,1) #get field in k-space
                        ifft_kF_k1 = np.zeros((len(Q_ij),N_side,N_side,N_side),dtype=dtype_r)
                        for j in range(len(kF_k)):
                            ifft_kF_k1[j] = FFTW_ifft(np.where(in_k1,kF_k[j]*MAS,0))
                        
                        #fpr the - x1 (x1.xi) parts
                        F_1,F_2  = Fields_func(delta,ki,xi,x_norm,1,1)
                        ifft_F_1 = np.zeros((len(F_1),N_side,N_side,N_side),dtype=dtype_r)
                        for j in range(len(F_1)):
                            ifft_F_1[j] = FFTW_ifft(np.where(in_k1,F_1[j]*MAS,0))

                        if rs[0] > 0:
                            #for r k.x2
                            ifft_F_Qk2 = np.zeros((len(Q_ij),N_side,N_side,N_side),dtype=dtype_r)
                            for j in range(len(Q_ij)):
                                ifft_F_Qk2[j] = FFTW_ifft(np.where(in_k2,Q_ij[j]*MAS,0))

                            Bk_lm += rs[0]*Amn_sum(N_bins,ifft_kF_k1,ifft_F_Qk2,ifft_F_k3)
                        
                            #for: - r x1 (x1.x2)   #similar to stuff above but with Fields func rather than Qpqrs
                            F_1,F_2  = Fields_func(delta,ki,xi,x_norm,1,1)

                            ifft_F_2 = np.zeros((len(F_1),N_side,N_side,N_side),dtype=dtype_r)
                            for j in range(len(F_1)):
                                ifft_F_2[j] = FFTW_ifft(np.where(in_k2,F_2[j]*MAS,0))

                            Bk_lm += -rs[0]*Amn_sum(N_bins,ifft_F_1,ifft_F_2,ifft_F_k3)
                        
                        if rs[1] > 0:
                            #for s k.x3              
                            #uses kF_k1 from before and Q_ij
                            ifft_F_Qk3 = np.zeros((N_bins,len(Q_ij),N_side,N_side,N_side),dtype=dtype_r)
                            for j in range(len(kF_k)):
                                for i in range(N_bins):
                                    ifft_F_Qk3[i,j] = FFTW_ifft(np.where(in_k3[i],Q_ij[j]*MAS,0))

                            Bk_lm += rs[1]*Amn_sum(N_bins,ifft_kF_k1,ifft_F_k2,ifft_F_Qk3,False)

                            #for -s x1 (x1.x3) #similar to before but binning is now with k3
                            ifft_F_3 = np.zeros((N_bins,len(F_1),N_side,N_side,N_side),dtype=dtype_r)
                            for j in range(len(kF_k)):
                                for i in range(N_bins):
                                    ifft_F_3[i,j] = FFTW_ifft(np.where(in_k3[i],F_2[j]*MAS,0))
                            
                            
                            Bk_lm += -rs[1]*Amn_sum(N_bins,ifft_F_1,ifft_F_k2,ifft_F_3,False)
                            
                    
                    if exorder == 2:#should be careful about bf.count... as we reuse stuff but should be ok as little is of second order...-but CHECK!!!
                        #this is going to be fucking long but here we go hope it is interesting!
                        #can't split r and s now...
                        
                        #for the r(1+r+s)(k. x2) parts--------------------------------------------
                        kF_k, Q_ij = Qpqrs2(delta,FFTW_fft(delta/x_norm),xi,ki,1) #get field in k-space
                        ifft_kF_k1 = np.zeros((len(Q_ij),N_side,N_side,N_side),dtype=dtype_r)
                        ifft_Q1_2 = np.zeros((len(Q_ij),N_side,N_side,N_side),dtype=dtype_r)
                        for j in range(len(kF_k)):
                            ifft_kF_k1[j] = FFTW_ifft(np.where(in_k1,kF_k[j]*MAS,0))
                            ifft_Q1_2[j] = FFTW_ifft(np.where(in_k2,Q_ij[j]*MAS,0))
                            
                        Bk_lm += r*(1+r+s)*Amn_sum(N_bins,ifft_kF_k1,ifft_Q1_2,ifft_F_k3)
                        
                        #for s(1+r+s) k.x3             
                        #uses kF_k1 from before and Q_ij
                        ifft_Q1_3 = np.zeros((N_bins,len(Q_ij),N_side,N_side,N_side),dtype=dtype_r)
                        for j in range(len(kF_k)):
                            for i in range(N_bins):
                                ifft_Q1_3[i,j] = FFTW_ifft(np.where(in_k3[i],Q_ij[j]*MAS,0))

                        Bk_lm += s*(1+r+s)*Amn_sum(N_bins,ifft_kF_k1,ifft_F_k2,ifft_Q1_3,False)
                        
                        #for  -r*(1+r+s) x1 (x1.x2) parts-------------------------------------------------
                        F_1,F_2 = Fields_func(delta,ki,xi,x_norm,1,1)
                        ifft_G1Q1_1 = np.zeros((len(F_1),N_side,N_side,N_side),dtype=dtype_r)
                        for j in range(len(F_1)):
                            ifft_G1Q1_1[j] = FFTW_ifft(np.where(in_k1,F_1[j]*MAS,0))

                        Bk_lm += -r*(1+r+s)*Amn_sum(N_bins,ifft_G1Q1_1,ifft_Q1_2,ifft_F_k3)          
                        
                        #for  -s*(1+r+s)*x1 (x1.x3) #similar to before but binning is now with k3
                        Bk_lm += -s*(1+r+s)*Amn_sum(N_bins,ifft_G1Q1_1,ifft_F_k2,ifft_Q1_3,False)
                               
                        #ok so now for x1 (xi.x1)^2    v.similar to x1 (x1.x2) parts---------------
                        #for (3/2)* r**2 x1 (x2.x1)^2
                        F_1,F_2  = Fields_func(delta,ki,xi,x_norm,1,2)
                        ifft_G1Q2_1 = np.zeros((len(F_1),N_side,N_side,N_side),dtype=dtype_r)
                        ifft_Q2_2 = np.zeros((len(F_1),N_side,N_side,N_side),dtype=dtype_r)
                        for j in range(len(F_1)):
                            ifft_G1Q2_1[j] = FFTW_ifft(np.where(in_k1,F_1[j]*MAS,0))
                            ifft_Q2_2[j] = FFTW_ifft(np.where(in_k2,F_2[j]*MAS,0)) 

                        Bk_lm += (3/2)* r**2*Amn_sum(N_bins,ifft_G1Q2_1,ifft_Q2_2,ifft_F_k3)          
                        
                        #for (3/2)* s**2 x1 (x1.x3)^2 #similar to before but binning is now with k3
                        ifft_Q2_3 = np.zeros((N_bins,len(F_1),N_side,N_side,N_side),dtype=dtype_r)
                        for j in range(len(F_1)):
                            for i in range(N_bins):
                                ifft_Q2_3[i,j] = FFTW_ifft(np.where(in_k3[i],F_2[j]*MAS,0))
                        
                        Bk_lm += (3/2)* s**2*Amn_sum(N_bins,ifft_G1Q2_1,ifft_F_k2,ifft_Q2_3,False)
                        
                        #ok now for xi (xi.x1)----------------------------------------------   
                        #for -r^2 x2 (x1.x2)
                        F_1,F_2  = Fields_funcbk(delta,ki,xi,x_norm,1,1) # returns a (3,3,grid.shape)
                        ifft_kQ1_1 = np.zeros((F_1.shape[0],F_1.shape[1],N_side,N_side,N_side),dtype=dtype_r)    
                        for i in range((F_1.shape[0])):
                            for j in range(F_1.shape[1]):
                                ifft_kQ1_1[i,j] = FFTW_ifft(np.where(in_k1,F_1[i,j]*MAS,0))
                        
                        #reshape into (9,grid.shape)
                        Bk_lm += -r**2 *Amn_sum(N_bins,ifft_kQ1_1.reshape((F_1.shape[0]*F_1.shape[1],N_side,N_side,N_side)),reshape69(ifft_Q2_2),ifft_F_k3)
                        
                        #for -s^2 x3 (x1.x3)   
                        #reshape into (N_bins,9,grid.shape)
                        Bk_lm += -s**2 *Amn_sum(N_bins,ifft_kQ1_1.reshape((F_1.shape[0]*F_1.shape[1],N_side,N_side,N_side)),ifft_F_k2,reshape69(ifft_Q2_3,False),False)
                        
                        #ok now for xi (xj.x1) parts----------------------------------------------
                        #so we will have 3 different fields - but we can just reuse previous shit
                        #for -r s x3(x1.x2)
                        Bk_lm += -r*s *Mixed_sum(N_bins,ifft_kQ1_1,ifft_Q1_2,ifft_Q1_3)
                        #for -r s x2(x1.x3)# same as above but kQ1 pairings are swapped!
                        Bk_lm += -r*s *Mixed_sum(N_bins,ifft_kQ1_1,ifft_Q1_2,ifft_Q1_3,False)
                         
                        #ok now for the x1 (xi.xi)--------------------------------------------------
                        #for -(1/2) r^2 x1 (x2.x2)
                        F_1 = Qpqrs(delta/x_norm**3,xi,ki,1)
                        ifft_q1_1 = FFTW_ifft(np.where(in_k1,F_1*MAS,0))
                        F_2 = FFTW_fft(delta*x_norm**2)
                        ifft_xx_2 = FFTW_ifft(np.where(in_k2,F_2*MAS,0))    
     
                        Bk_lm += -(1/2)*r**2 *Amn_sum(N_bins,ifft_q1_1[np.newaxis,...],ifft_xx_2[np.newaxis,...],ifft_F_k3)
                        #for -(1/2) s^2 x1 (x3.x3)
                        ifft_xx_3 = np.zeros((N_bins,N_side,N_side,N_side),dtype=dtype_r)    
                        for i in range(N_bins):
                            ifft_xx_3[i] = FFTW_ifft(np.where(in_k3[i],F_2*MAS,0))
                            
                        Bk_lm += -(1/2)*s**2 *Amn_sum(N_bins,ifft_q1_1[np.newaxis,...],ifft_F_k2[np.newaxis,...],ifft_xx_3)
    
                        #ok now for -rs x1 (x2.x3)-------------------------------------------------------
                        Bk_lm += - r*s*Amn_sum_x2x3(N_bins,ifft_q1_1,ifft_Q1_2,ifft_Q1_3)
                            
                        #ok now for the final large boi---------------------------------------------------
                        #3rs (k.x1)(x2.x1)(x3.x1)
                        F_1 = Fields_funcbk3(delta,ki,xi,x_norm,1,1)
                        ifft_GQQ_1 = np.zeros((F_1.shape[0],F_1.shape[1],N_side,N_side,N_side),dtype=dtype_r)    
                        for i in range((F_1.shape[0])):
                            for j in range(F_1.shape[1]):
                                ifft_GQQ_1[i,j] = FFTW_ifft(np.where(in_k1,F_1[i,j]*MAS,0))
                            
                        Bk_lm += 3*r*s *Mixed_sum(N_bins,ifft_GQQ_1,ifft_Q1_2,ifft_Q1_3)
                            
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
            self.mono =  Bk_endpoint_theta(delta,0)
            if l != 0:
                self.bk = Bk_theta(delta,l) 
        
        else:
            # this is to pick which estimator function to call! - based on LOS and l
            #self.mono = 0
            if  rs == [0,0]:
                self.bk = Bk_endpoint_theta(delta,l)
            else:
                self.bk = Bk_theta(delta,l)
