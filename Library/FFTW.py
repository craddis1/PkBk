import numpy as np
import pyfftw

#required to set variable probably
FFT_number = 0
iFFT_number = 0
#does iFFT using FFTW
def FFTW_ifft_single(delta,threads=1):
    global iFFT_number
    iFFT_number += 1
    Nside = len(delta)
    in_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.complex64)
    out_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.complex64)
    fftw_obj = pyfftw.FFTW(in_array,out_array,axes=(0,1,2),flags=("FFTW_ESTIMATE", ),direction='FFTW_BACKWARD',threads=threads)
    in_array[:] = delta
    return fftw_obj(delta)#np.fft.ifftn(delta)

#does FFT using FFTW - to single np.complex64 precision
def FFTW_fft_single(delta,threads=1):
    global FFT_number
    FFT_number += 1
    Nside = len(delta)
    in_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.complex64)
    out_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.complex64)
    #print("size:", round(out_array.nbytes / 1000 / 1000), "MB")
    fft_obj=pyfftw.FFTW(in_array,out_array,axes=(0,1,2),flags=("FFTW_ESTIMATE", ),threads=threads)
    in_array[:] = delta
    return fft_obj(delta)

#does iFFT using FFTW
def FFTW_ifft_double(delta,threads=1):
    global iFFT_number
    iFFT_number += 1
    Nside = len(delta)
    in_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.complex128)
    out_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.complex128)
    fftw_obj = pyfftw.FFTW(in_array,out_array,axes=(0,1,2),flags=("FFTW_ESTIMATE", ),direction='FFTW_BACKWARD',threads=threads)
    in_array[:] = delta
    return fftw_obj(delta)#np.fft.ifftn(delta)

#does FFT using FFTW - to single np.complex64 precision
def FFTW_fft_double(delta,threads=1):
    global FFT_number
    FFT_number += 1
    Nside = len(delta)
    in_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.complex128)
    out_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.complex128)
    #print("size:", round(out_array.nbytes / 1000 / 1000), "MB")
    fft_obj=pyfftw.FFTW(in_array,out_array,axes=(0,1,2),flags=("FFTW_ESTIMATE", ),threads=threads)
    in_array[:] = delta
    return fft_obj(delta)

def FFTW_rfft_single(delta,threads=1):
    global FFT_number
    FFT_number += 1
    # align arrays
    Nside = len(delta)
    in_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.float32)
    out_array = pyfftw.empty_aligned((Nside,Nside,Nside//2 + 1), dtype=np.complex64)
    fftw_obj = pyfftw.FFTW(in_array,out_array,axes=(0,1,2),flags=("FFTW_ESTIMATE", ),threads=threads)
    in_array[:] = delta
    return fftw_obj(in_array)#maybe try fftw_obj(in_array,out_array)

# This function performs the 3D FFT of a field in double precision
def FFTW_irfft_single(delta,threads=1):#IFFT3Dr_d
    global iFFT_number
    iFFT_number += 1
    # align arrays
    Nside = len(delta)
    in_array = pyfftw.empty_aligned((Nside,Nside,Nside//2 + 1), dtype=np.complex64)
    out_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.float32)
    fftw_obj = pyfftw.FFTW(in_array,out_array,axes=(0,1,2),flags=("FFTW_ESTIMATE", ),direction='FFTW_BACKWARD',threads=threads)
    in_array[:] = delta
    return fftw_obj(in_array)#maybe try fftw_obj(in_array,out_array)

# Does 3D real (r)FFTs with FFTW using double (np.complex128) precision
def FFTW_rfft_double(delta,threads=1):
    global FFT_number
    FFT_number += 1
    # align arrays
    Nside = len(delta)
    in_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.float64)
    out_array = pyfftw.empty_aligned((Nside,Nside,Nside//2 + 1), dtype=np.complex128)
    fftw_obj = pyfftw.FFTW(in_array,out_array,axes=(0,1,2),flags=("FFTW_ESTIMATE", ),threads=threads)
    in_array[:] = delta
    return fftw_obj(in_array)#maybe try fftw_obj(in_array,out_array)

# Does 3D inverse real (i)(r)FFTs with FFTW using double (np.complex128) precision
def FFTW_irfft_double(delta,threads=1):#IFFT3Dr_d
    global iFFT_number
    iFFT_number += 1
    # align arrays
    Nside = len(delta)
    in_array = pyfftw.empty_aligned((Nside,Nside,Nside//2 + 1), dtype=np.complex128)
    out_array = pyfftw.empty_aligned((Nside,Nside,Nside), dtype=np.float64)
    fftw_obj = pyfftw.FFTW(in_array,out_array,axes=(0,1,2),flags=("FFTW_ESTIMATE", ),direction='FFTW_BACKWARD',threads=threads)
    in_array[:] = delta
    return fftw_obj(in_array)#maybe try fftw_obj(in_array,out_array)