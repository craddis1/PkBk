@jit(nopython=True,fastmath=fastmath,parallel=True)#
        def ifft_sum(field1: complex64[:, :], field2: complex64[:, :], field3: complex64[:, :]) -> complex64[:, :, :]: # does sum over fields where there are possible closed triangles...      
            Bk_lm = np.zeros((N_bins,N_bins,N_bins),dtype=np.complex64)
            for i in prange(N_bins):
                for j in range(N_bins):       #we remove these constraints as it is not symmetric for l>0
                    #if i > 2*j:
                        #continue                              #these values should all be 0 anyway as dirac delta
                    for k in range(N_bins):
                        #if (i > j+k) or ((j > i+k)or (k > i+j)):
                            #continue
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
                for j in prange(N_bins):# i+1 -> k1 >= k2       #we remove these constraints as it is not symmetric for l>0
                    for k in prange(N_bins):#j+1 -> k2 >= k3              
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
                for j in range(N_bins):# i+1 -> k1 >= k2       #we remove these constraints as it is not symmetric for l>0
                    for k in range(N_bins):#j+1 -> k2 >= k3              
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
                for j in range(N_bins):# i+1 -> k1 >= k2       #we remove these constraints as it is not symmetric for l>0
                    for k in range(N_bins):#j+1 -> k2 >= k3              
                        sum_real = np.float32(0.0)
                        sum_imag = np.float32(0.0)
                        for m in range(field2.shape[1]):
                            for idx in range(field1.shape[-1]):
                                sum_real = sum_real + field1.real[i, idx] * field2.real[j,m, idx] * field3.real[k,m, idx] - field1.imag[i, idx] * field2.imag[j,m, idx] * field3.real[k,m, idx] - field1.real[i, idx] * field2.imag[j,m, idx] * field3.imag[k,m, idx] - field1.imag[i, idx] * field2.real[j,m, idx] * field3.imag[k,m, idx]
                                sum_imag = sum_imag + field1.real[i, idx] * field2.imag[j,m, idx] * field3.real[k,m, idx] + field1.imag[i, idx] * field2.real[j,m, idx] * field3.real[k,m, idx] + field1.real[i, idx] * field2.real[j,m, idx] * field3.imag[k,m, idx] - field1.imag[i, idx] * field2.imag[j,m, idx] * field3.imag[k,m, idx]
                        Bk_lm[i, j, k] = sum_real + 1j*sum_imag
            return Bk_lm