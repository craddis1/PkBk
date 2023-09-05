# peturbative expansion for LOS
import numpy as np
#from time import time # maybe add computation time feature
from itertools import combinations_with_replacement
import pickle
from numba import jit
import sys
sys.path.append('Library')
import base_funcs as bf

class Pk:
    def __init__(self,delta,L,Nside,l,k,grid_info,binning_info,
                 t=0,ex_order=1,iFFT=False,dtype=np.complex128,threads=1,rfft=False,verbose=True):
        
        #unpack variables from grid info and binning info...
        xi,x_norm,ki,k_mag,MAS,k_f,k_ny = grid_info
        In_bin,N_modes = binning_info
        
        Npix = Nside**3
        V = L**3
        H = V/Npix
        const = (H**2)/V
        
        #raise warning if bad dtype
        if dtype != np.complex128 and dtype != np.complex64:
            raise Exception("Invalid dtype")
            
        N_side = len(delta)
        #import which fft type we want to use for the estimator
        #also sets some values in the case of real FFTs - i.e. fourier space arrays are halved
        if rfft:
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
        
        #initialize global variables
        #FFT_number=0;iFFT_number=0
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
        
        #Caluclate Q_ij etc and sums over to get G - without summing Qpqrs2 (i.e.  requires a lot more memory...
        #so this is for the q1.r1 parts
        def Qpqrs(delta,xi,ki,l): # extends
            if l == 0:
                return FFTW_fft(delta) # this is need for the way Pk_bisector is implemented

            kQ_sum = np.zeros((N_side,N_side,N_side_r)) # create empty array

            ind = np.array(list(combinations_with_replacement(np.arange(3), l))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
            for i in ind: # so iterates for each possible ij etc

                k_prod = np.prod(ki[i],axis=0) # sum kx etc arrays for each combination
                x_prod = np.prod(xi[i],axis=0) #real parts
                #count distinct permutation for each combination
                kQ_sum = np.add(kQ_sum,np.multiply(bf.count_distinct_permutations(i)*k_prod,FFTW_fft(delta*x_prod),dtype=dtype)) # e.g. all terms in square brackets eq.20 scoccimarro

            return kQ_sum

        #this is pretty nice and beefy function used when we have powers of the dot products...
        # (k1.x_{field})^l1 (x1.x2)^l2
        def Fields_func(delta,k1,xi,x_norm,l1,l2,field=1):
            #for the two cartesian expansions
            ind1 = np.array(list(combinations_with_replacement(np.arange(3), l1))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
            ind2 = np.array(list(combinations_with_replacement(np.arange(3), l2))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
   
            F_1 = np.zeros((len(ind2),N_side,N_side,N_side_r),dtype=np.complex128) # create empty array
            F_2 = np.zeros((len(ind2),N_side,N_side,N_side_r),dtype=np.complex128) # create empty array

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
                    F_1[i] += bf.count_distinct_permutations(j)*k_prod*FFTW_fft(delta1*x_prod*x_prod2) # e.g. all terms in square brackets eq.20 scoccimarro
                    
            return F_1,F_2
        
        # (k1.x_2)^l1 (k1.x_1)^l1 (x1.x2)^l2 # this for one term in second order quadrupole expansion
        def Fields_func2(delta,k1,xi,x_norm,l1,l2):
            #for the two cartesian expansions
            ind1 = np.array(list(combinations_with_replacement(np.arange(3), l1))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
            ind2 = np.array(list(combinations_with_replacement(np.arange(3), l2))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
   
            F_1 = np.zeros((len(ind2),N_side,N_side,N_side_r),dtype=np.complex128) # create empty array
            F_2 = np.zeros((len(ind2),N_side,N_side,N_side_r),dtype=np.complex128) # create empty array

            delta1 = delta/(x_norm**(2*l1+2*l2))  #assign fields
            
            # so iterates for each possible ij etc   
            for i in range(len(ind2)):  # this for x1.x2 part
                x_prod2 = np.prod(xi[ind2[i]],axis=0)
                for j in ind1: #for k1.x1 and k1.x2 part
                    k_prod = np.prod(k1[j],axis=0) # sum kx etc arrays for each combination
                    x_prod = np.prod(xi[j],axis=0) #sum xi etc
                    perms1 = bf.count_distinct_permutations(j)
                    perms2 = bf.count_distinct_permutations(ind2[i])
                    F_1[i] += perms1*k_prod*FFTW_fft(delta1*x_prod*x_prod2) 
                    F_2[i] += perms1*perms2*k_prod*FFTW_fft(delta*x_prod*x_prod2) 
                    
            return F_1,F_2
        
        # equivalent of power_bin = const*np.sum(power_k[np.newaxis, ...] * In_bin, axis=(1,2,3))/N_modes
        def sum_loop(F_1,F_2,composite=False):
            """Does sum loop over two fields - if composite = True: then it does extra loop over them fields"""
            power_k = F_1*MAS*np.conj(F_2*MAS)#is conjugate as F(-k) = F*(k)
            Pk_lm_empty = np.zeros(len(In_bin),dtype=dtype)
            if composite != True:
                for j in range(len(In_bin)):
                    power_bin = np.sum(power_k[In_bin[j]])
                    Pk_lm_empty[j] += const*power_bin/N_modes[j]
                    
            else:
                for i in range(len(F_1)):
                    for j in range(len(In_bin)):
                        power_bin = np.sum(power_k[i][In_bin[j]])
                        Pk_lm_empty[j] += const*power_bin/N_modes[j]
            
            return Pk_lm_empty


        #calculates Pk using a direct estimator method 
        def Pk_main(delta,k_,l):
            global FFT_number
            FFT_number = 0

            Nbins = len(k_)

            def main_func(l):
                """does main calculation for powerspectrum multipoles with different LOS"""
                if l == 0:
                    return Pk_mono
                
                #get the two fields G_l etc - for second field k2 = -k1 which we are expanding around    
                F_1 = Qpqrs(delta/(x_norm**l),xi,ki,l)
                F_2 = FFTW_fft(delta)  
                Pk_lm = sum_loop(F_1,F_2)

                if t > 0:
                    if l==1:
                        if ex_order == 1:
                            #t k.x2/x1
                            F_1 = FFTW_fft(delta/x_norm)#Qpqrs(delta,norm,ki,1)
                            F_2 = Qpqrs(delta,xi,ki,1)#so this has the k term...
                            Pk_lm += t*sum_loop(F_1,F_2)
                            
                            #ok so -t(k1.x1)(x1.x2) #so there are 9 terms -(first 3 are collasped down)...
                            F_1,F_2 = Fields_func(delta,ki,xi,x_norm,1,1)
                            Pk_lm += -t*sum_loop(F_1,F_2,True)#ok F_1 and F_2 have shape (3,(field))

                        elif ex_order == 2:
                            #lets go to second order...
                            #only thing that changes for 1st few terms is (t**2+t)
                            #(t^2+t) k.x2/x1
                            F_1 = FFTW_fft(delta/x_norm)#Qpqrs(delta,norm,ki,1)
                            F_2 = Qpqrs(delta,xi,ki,1)#so this has the k term...
                            Pk_lm += (t**2+t)*sum_loop(F_1,F_2)
                            
                            
                            #ok so -(t^2+t)(k1.x1)(x1.x2) #so there are 9 terms -(first 3 are collasped down)...
                            F_1,F_2 = Fields_func(delta,ki,xi,x_norm,1,1)
                            Pk_lm += -(t**2+t)*sum_loop(F_1,F_2,True)#ok F_1 and F_2 have shape (3,(field))
                             
                            
                            #-(t^2)(k1.x2)(x1.x2) #!!!! isn't this superflous 
                            F_2,F_1 = Fields_func(delta,ki,xi,x_norm,1,1,2) # when field =2 then F_2 and F_1 switch
                            Pk_lm += -(t**2)*sum_loop(F_1,F_2,True)#ok F_1 and F_2 have shape (3,(field)) 
                            
                                    
                            #3/2 (t^2)(k1.x1)(x1.x2)^2
                            F_1,F_2 = Fields_func(delta,ki,xi,x_norm,1,2)
                            Pk_lm += (3/2) *(t**2)*sum_loop(F_1,F_2,True)#ok F_1 and F_2 have shape (6,(field))
                                            
                            #-1/2 (t^2)(k1.x1)(x2.x2)
                            F_1 = Qpqrs(delta/x_norm**3,xi,ki,1)#so this has the k term...
                            F_2 = FFTW_fft(delta*x_norm**2)
                            Pk_lm += -(1/2) *(t**2)*sum_loop(F_1,F_2)
                                          
                    if l==2:                   
                        if ex_order ==1:
                            #2t (k.x1)(k.x2)/x1
                            F_1 = Qpqrs(delta/(x_norm**2),xi,ki,1)
                            F_2 = Qpqrs(delta,xi,ki,1)
                            Pk_lm += 2*t*sum_loop(F_1,F_2)

                            #-2t(k.x1)^2 (x1.x2)/x2
                            F_1,F_2 = Fields_func(delta,ki,xi,x_norm,2,1)
                            Pk_lm += -2*t*sum_loop(F_1,F_2,True)
                                    
                        elif ex_order == 2:
                            
                            #lets go to second order...
                            #2(t^2+t) (k.x1)(k.x2)/x1
                            F_1 = Qpqrs(delta/x_norm**2,xi,ki,1)
                            F_2 = Qpqrs(delta,xi,ki,1)
                            Pk_lm += 2*(t**2+t)*sum_loop(F_1,F_2)

                            #-2(t^2+t)(k.x1)^2 (x1.x2)/x2
                            F_1,F_2 = Fields_func(delta,ki,xi,x_norm,2,1)
                            Pk_lm += -2*(t**2+t)*sum_loop(F_1,F_2,True)

                            #t^2 (k.x2)^2
                            F_1 = FFTW_fft(delta/x_norm**2)  
                            F_2 = Qpqrs(delta,xi,ki,2)
                            Pk_lm += t**2 *sum_loop(F_1,F_2)

                            #-4 t^2 (k.x1)(k.x2)(x1.x2)    
                            F_1,F_2 = Fields_func2(delta,ki,xi,x_norm,1,1)
                            Pk_lm += -4*t**2 *sum_loop(F_1,F_2,True)

                            #4 (t^2) (k.x1)^2 (x1.x2)^2
                            l1=2;l2=2
                            F_1,F_2 = Fields_func(delta,ki,xi,x_norm,l1,l2)
                            Pk_lm += 4*t**2 *sum_loop(F_1,F_2,True)

                            #-(t^2)(k1.x1)^2 (x2.x2)
                            F_1 = Qpqrs(delta/x_norm**4,xi,ki,2)#so this has the k term...
                            F_2 = FFTW_fft(delta*x_norm**2)
                            Pk_lm += -t**2 *sum_loop(F_1,F_2)
                          
                return Pk_lm

            def Pk_legendre(l):
                if l == 0:
                    return self.mono #main_func(0)
                if l == 1:
                    return main_func(1)
                if l == 2:
                    return (1/2)*(3*main_func(2)-main_func(0))
                if l == 3:
                    return (1/2)*(5*main_func(3)- 3*main_func(1))
                if l == 4:
                    return (1/8)*(35*main_func(4)- 30*main_func(2) + 3*main_func(0))
                else:
                    raise "Multipole not implemented"

            Pk_lm = Pk_legendre(l)
            return (2*l+1)*(Pk_lm)
        
        #calculates Pk using a direct estimator method - endpoint version
        def Pk_endpoint(delta,k,l):
            global FFT_number
            FFT_number = 0
            
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
                    
            delta_k = FFTW_fft(delta)
            
            #calculate convolved field.
            delta_l = delta_lm(delta,delta_k,xi/x_norm,ki,l)*MAS #this is the delta_l #plane_parallel
            delta_kk = delta_k*MAS

            Npix = Nside**3
            V = L**3
            H = V/Npix
            const = (H**2)/V

            Pk = np.zeros(len(k),dtype=dtype)
            power_k = delta_l*np.conj(delta_kk)#has to be conjugate as this creates the other part from from rfft - so if you consider full fft then we get double N_modes and Power_bin which cancels
            for i in range(len(k)):
                power_bin = np.sum(power_k[In_bin[i]])
                Pk[i] = const*power_bin/N_modes[i]
            return (2*l+1)*Pk

        #if even then calculate monopole    
        if l % 2 == 0:
            # just calculate the monopole - it useful
            Pk_mono = Pk_endpoint(delta,k,0)
            self.mono = Pk_mono
            self.pk = Pk_mono
            if l != 0:
                self.pk = Pk_main(delta,k,l)
        
        else:
            # this is to pick which estimator function to call! - based on LOS and l
            #self.mono = 0
            self.pk = Pk_main(delta,k,l)
                    
                
         