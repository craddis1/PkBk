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
    def __init__(self,delta,L,k,grid_info,binning_info,
                 iFFT=False,dtype=np.complex128,threads=1,rfft=False,verbose=True):
        
        #unpack variables from grid info and binning info...
        self.xi,self.x_norm,self.ki,self.k_mag,self.MAS,self.k_f,self.k_ny = grid_info
        self.In_bin,self.N_modes = binning_info
        self.Nbins = len(k)
        
        self.N_side = self.k_mag.shape[0]
        self.Npix = self.N_side**3
        self.V = L**3
        self.H = self.V/self.Npix
        self.const = (self.H**2)/self.V
        
        # Raise warning if bad dtype
        if dtype != np.complex128 and dtype != np.complex64:
            raise Exception("Invalid dtype")
             
        # Import the correct FFT and iFFT functions based on the dtype and rfft flag
        # for real FFTs fourier space arrays are halved
        if rfft:
            self.N_side_r = self.N_side // 2 + 1
            if dtype == np.complex128:
                from FFTW import FFTW_irfft_double as iFFT_
                from FFTW import FFTW_rfft_double as FFT_       
                self.dtype_r = np.float64
            elif dtype == np.complex64:
                from FFTW import FFTW_irfft_single as iFFT_
                from FFTW import FFTW_rfft_single as FFT_
                self.dtype_r = np.float32
        else:
            if dtype == np.complex128:
                from FFTW import FFTW_ifft_double as iFFT_
                from FFTW import FFTW_fft_double as FFT_         
            elif dtype == np.complex64:
                from FFTW import FFTW_ifft_single as iFFT_
                from FFTW import FFTW_fft_single as FFT_
            self.dtype_r = dtype                         
            self.N_side_r = self.N_side
        
        self.dtype = dtype

        self.threads = threads # number of threads for fft
        self.FFT_ = FFT_
        self.iFFT_ = iFFT_
        self.FFT_number = 0
        self.iFFT_number = 0

    #wrapper function for FFT scheme imported from FFTW module - counts FFTs
    def FFTW_fft(self,delta):
        
        self.FFT_number += 1
        return self.FFT_(delta,self.threads)
    #wrapper function for iFFT scheme imported from FFTW module - counts iFFTs
    def FFTW_ifft(self,delta):
        
        self.iFFT_number += 1
        return self.iFFT_(delta,self.threads)
        
    #Caluclate Q_ij etc and sums over to get G - without summing Qpqrs2 (i.e.  requires a lot more memory...
    #so this is for the q1.r1 parts
    def Qpqrs(self,delta,xi,ki,l): # extends
        if l == 0:
            return self.FFTW_fft(delta) # this is need for the way Pk_bisector is implemented

        kQ_sum = np.zeros((self.N_side,self.N_side,self.N_side_r)) # create empty array

        ind = np.array(list(combinations_with_replacement(np.arange(3), l))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
        for i in ind: # so iterates for each possible ij etc

            k_prod = np.prod(ki[i],axis=0) # sum kx etc arrays for each combination
            x_prod = np.prod(xi[i],axis=0) #real parts
            #count distinct permutation for each combination
            kQ_sum = np.add(kQ_sum,np.multiply(bf.count_distinct_permutations(i)*k_prod,self.FFTW_fft(delta*x_prod),dtype=self.dtype)) # e.g. all terms in square brackets eq.20 scoccimarro

        return kQ_sum

    #this is pretty nice and beefy function used when we have powers of the dot products...
    # (k1.x_{field})^l1 (x1.x2)^l2
    def Fields_func(self,delta,k1,xi,x_norm,l1,l2,field=1):
        #for the two cartesian expansions
        ind1 = np.array(list(combinations_with_replacement(np.arange(3), l1))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
        ind2 = np.array(list(combinations_with_replacement(np.arange(3), l2))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components

        F_1 = np.zeros((len(ind2),self.N_side,self.N_side,self.N_side_r),dtype=np.complex128) # create empty array
        F_2 = np.zeros((len(ind2),self.N_side,self.N_side,self.N_side_r),dtype=np.complex128) # create empty array

        delta1 = delta*1  #assign fields
        delta2 = delta*1
        if field == 2:
            delta2 *= 1/(x_norm**(l1+2*l2))
        else:
            delta1 *= 1/(x_norm**(l1+2*l2))

        # so iterates for each possible ij etc   
        for i in range(len(ind2)):  # this for x1.x2 part
            x_prod2 = np.prod(xi[ind2[i]],axis=0)
            F_2[i] = bf.count_distinct_permutations(ind2[i])*self.FFTW_fft(delta2*x_prod2) # e.g. all terms in square brackets eq.20 scoccimarro
            for j in ind1: #for k1.x1 part #this is then condensed...
                k_prod = np.prod(k1[j],axis=0) # sum kx etc arrays for each combination
                x_prod = np.prod(xi[j],axis=0) #sum xi etc
                F_1[i] += bf.count_distinct_permutations(j)*k_prod*self.FFTW_fft(delta1*x_prod*x_prod2) # e.g. all terms in square brackets eq.20 scoccimarro

        return F_1,F_2

    # (k1.x_2)^l1 (k1.x_1)^l1 (x1.x2)^l2 # this for one term in second order quadrupole expansion
    def Fields_func2(self,delta,k1,xi,x_norm,l1,l2):
        #for the two cartesian expansions
        ind1 = np.array(list(combinations_with_replacement(np.arange(3), l1))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components
        ind2 = np.array(list(combinations_with_replacement(np.arange(3), l2))) #find all combinations for l components from x,y,z - (l+1)(l+2)/2 components

        F_1 = np.zeros((len(ind2),self.N_side,self.N_side,self.N_side_r),dtype=np.complex128) # create empty array
        F_2 = np.zeros((len(ind2),self.N_side,self.N_side,self.N_side_r),dtype=np.complex128) # create empty array

        delta1 = delta/(x_norm**(2*l1+2*l2))  #assign fields

        # so iterates for each possible ij etc   
        for i in range(len(ind2)):  # this for x1.x2 part
            x_prod2 = np.prod(xi[ind2[i]],axis=0)
            for j in ind1: #for k1.x1 and k1.x2 part
                k_prod = np.prod(k1[j],axis=0) # sum kx etc arrays for each combination
                x_prod = np.prod(xi[j],axis=0) #sum xi etc
                perms1 = bf.count_distinct_permutations(j)
                perms2 = bf.count_distinct_permutations(ind2[i])
                F_1[i] += perms1*k_prod*self.FFTW_fft(delta1*x_prod*x_prod2) 
                F_2[i] += perms1*perms2*k_prod*self.FFTW_fft(delta*x_prod*x_prod2) 

        return F_1,F_2

    # equivalent of power_bin = const*np.sum(power_k[np.newaxis, ...] * In_bin, axis=(1,2,3))/N_modes
    def sum_loop(self,F_1,F_2,composite=False):
        """Does sum loop over two fields - if composite = True: then it does extra loop over them fields"""
        N_modes = self.N_modes
        In_bin = self.In_bin
        
        power_k = F_1*self.MAS*np.conjugate(F_2*self.MAS)#is conjugate as F(-k) = F*(k)
        
        Pk_lm_empty = np.zeros(len(In_bin),dtype=self.dtype)
        if composite != True:
            for j in range(len(In_bin)):
                power_bin = np.sum(power_k[In_bin[j]])
                Pk_lm_empty[j] += self.const*power_bin/N_modes[j]

        else:
            for i in range(len(F_1)):
                for j in range(len(In_bin)):
                    power_bin = np.sum(power_k[i][In_bin[j]])
                    Pk_lm_empty[j] += self.const*power_bin/N_modes[j]

        return Pk_lm_empty


    #calculates Pk using a direct estimator method 
    def Pk_main(self,delta,l,exorder=0,t=0,delta2=[0]):
        if len(delta2)==1: #for single tracer
            delta2=delta
            
        def main_func(l):
            """does main calculation for powerspectrum multipoles with different LOS"""
            if l == 0:
                return self.Pk_mono

            #get the two fields G_l etc - for second field k2 = -k1 which we are expanding around    
            F_1 = self.Qpqrs(delta/(self.x_norm**l),self.xi,self.ki,l)
            F_2 = self.FFTW_fft(delta2)  
            Pk_lm = self.sum_loop(F_1,F_2)

            if t > 0:
                xi = self.xi
                ki = self.ki
                x_norm = self.x_norm
                if l==1:
                    if ex_order == 1:
                        #t k.x2/x1
                        F_1 = self.FFTW_fft(delta/x_norm)#Qpqrs(delta,norm,ki,1)
                        F_2 = self.Qpqrs(delta,xi,ki,1)#so this has the k term...
                        Pk_lm += t*self.sum_loop(F_1,F_2)

                        #ok so -t(k1.x1)(x1.x2) #so there are 9 terms -(first 3 are collasped down)...
                        F_1,F_2 = self.Fields_func(delta,ki,xi,x_norm,1,1)
                        Pk_lm += -t*self.sum_loop(F_1,F_2,True)#ok F_1 and F_2 have shape (3,(field))

                    elif ex_order == 2:
                        #lets go to second order...
                        #only thing that changes for 1st few terms is (t**2+t)
                        #(t^2+t) k.x2/x1
                        F_1 = self.FFTW_fft(delta/x_norm)#Qpqrs(delta,norm,ki,1)
                        F_2 = self.Qpqrs(delta,xi,ki,1)#so this has the k term...
                        Pk_lm += (t**2+t)*self.sum_loop(F_1,F_2)


                        #ok so -(t^2+t)(k1.x1)(x1.x2) #so there are 9 terms -(first 3 are collasped down)...
                        F_1,F_2 = self.Fields_func(delta,ki,xi,x_norm,1,1)
                        Pk_lm += -(t**2+t)*self.sum_loop(F_1,F_2,True)#ok F_1 and F_2 have shape (3,(field))


                        #-(t^2)(k1.x2)(x1.x2) #!!!! isn't this superflous 
                        F_2,F_1 = self.Fields_func(delta,ki,xi,x_norm,1,1,2) # when field =2 then F_2 and F_1 switch
                        Pk_lm += -(t**2)*self.sum_loop(F_1,F_2,True)#ok F_1 and F_2 have shape (3,(field)) 


                        #3/2 (t^2)(k1.x1)(x1.x2)^2
                        F_1,F_2 = self.Fields_func(delta,ki,xi,x_norm,1,2)
                        Pk_lm += (3/2) *(t**2)*self.sum_loop(F_1,F_2,True)#ok F_1 and F_2 have shape (6,(field))

                        #-1/2 (t^2)(k1.x1)(x2.x2)
                        F_1 = self.Qpqrs(delta/x_norm**3,xi,ki,1)#so this has the k term...
                        F_2 = self.FFTW_fft(delta*x_norm**2)
                        Pk_lm += -(1/2) *(t**2)*self.sum_loop(F_1,F_2)

                if l==2:                   
                    if ex_order ==1:
                        #2t (k.x1)(k.x2)/x1
                        F_1 = self.Qpqrs(delta/(x_norm**2),xi,ki,1)
                        F_2 = self.Qpqrs(delta,xi,ki,1)
                        Pk_lm += 2*t*self.sum_loop(F_1,F_2)

                        #-2t(k.x1)^2 (x1.x2)/x2
                        F_1,F_2 = self.Fields_func(delta,ki,xi,x_norm,2,1)
                        Pk_lm += -2*t*self.sum_loop(F_1,F_2,True)

                    elif ex_order == 2:

                        #lets go to second order...
                        #2(t^2+t) (k.x1)(k.x2)/x1
                        F_1 = self.Qpqrs(delta/x_norm**2,xi,ki,1)
                        F_2 = self.Qpqrs(delta,xi,ki,1)
                        Pk_lm += 2*(t**2+t)*self.sum_loop(F_1,F_2)

                        #-2(t^2+t)(k.x1)^2 (x1.x2)/x2
                        F_1,F_2 = self.Fields_func(delta,ki,xi,x_norm,2,1)
                        Pk_lm += -2*(t**2+t)*self.sum_loop(F_1,F_2,True)

                        #t^2 (k.x2)^2
                        F_1 = self.FFTW_fft(delta/x_norm**2)  
                        F_2 = self.Qpqrs(delta,xi,ki,2)
                        Pk_lm += t**2 *self.sum_loop(F_1,F_2)

                        #-4 t^2 (k.x1)(k.x2)(x1.x2)    
                        F_1,F_2 = self.Fields_func2(delta,ki,xi,x_norm,1,1)
                        Pk_lm += -4*t**2 *self.sum_loop(F_1,F_2,True)

                        #4 (t^2) (k.x1)^2 (x1.x2)^2
                        l1=2;l2=2
                        F_1,F_2 = self.Fields_func(delta,ki,xi,x_norm,l1,l2)
                        Pk_lm += 4*t**2 *self.sum_loop(F_1,F_2,True)

                        #-(t^2)(k1.x1)^2 (x2.x2)
                        F_1 = self.Qpqrs(delta/x_norm**4,xi,ki,2)#so this has the k term...
                        F_2 = self.FFTW_fft(delta*x_norm**2)
                        Pk_lm += -t**2 *self.sum_loop(F_1,F_2)

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
    def Pk_endpoint(self,delta,l,delta2=[0]):
        if len(delta2)==1:
            delta2= delta
        
        N_modes = self.N_modes
        In_bin = self.In_bin

        #Lengendre multipoles
        def delta_lm(delta_x,delta_k,xi,ki,l):  #convolves delta with legendre polynomial
            if l == 0:
                return delta_k
            if l==1:
                Q_x = self.Qpqrs(delta_x,xi,ki,l)
                return Q_x
            if l==2:
                Q_xx = self.Qpqrs(delta_x,xi,ki,l)
                return (3/2)*Q_xx - (1/2)*delta_k
            if l==3:
                Q_xx = self.Qpqrs(delta_x,xi,ki,l)
                return (1/2)*(5*Q_xx - 3*delta_lm(delta_x,delta_k,xi,ki,1))
            if l==4:
                Q_xxxx = self.Qpqrs(delta_x,xi,ki,l)
                return (35/8)*Q_xxxx - (5/2)*delta_lm(delta_x,delta_k,xi,ki,2) - (7/8)*delta_k
            else:
                raise Exception(l,"l-multipole not implemented")

        delta_k = self.FFTW_fft(delta)

        #calculate convolved field.
        delta_l = delta_lm(delta,delta_k,self.xi/self.x_norm,self.ki,l)*self.MAS #this is the delta_l #plane_parallel
        delta_kk = self.FFTW_fft(delta2)*self.MAS

        Npix = self.Npix
        V = self.V
        H = self.H
        const = self.const

        Pk = np.zeros(self.Nbins,dtype=self.dtype)
        power_k = delta_l*np.conjugate(delta_kk)#has to be conjugate as this creates the other part from from rfft - so if you consider full fft then we get double N_modes and Power_bin which cancels
        for i in range(self.Nbins):
            power_bin = np.sum(power_k[In_bin[i]])
            Pk[i] = const*power_bin/N_modes[i]
        return (2*l+1)*Pk

    def get_Pk(self,delta,l=0,exorder=0,t=0,delta2=[0]):
        #if even then calculate monopole    
        if l % 2 == 0:
            if not hasattr(self, 'mono'):#if uncalculated - calculate
                Pk_mono = self.Pk_endpoint(delta,0,delta2=delta2)
                self.Pk_mono = Pk_mono
                self.pk = Pk_mono

            if l != 0:
                self.pk = self.Pk_main(delta,l,exorder=exorder,t=t,delta2=delta2)

        else:
            # this is to pick which estimator function to call! - based on LOS and l
            self.pk = self.Pk_main(delta,l,exorder=exorder,t=t,delta2=delta2)
            
        return self.pk
                    
                
         