
from numba import jit
#this is just some functions which are throughout main class - but are relativaley mundane but useful and so are put here to neaten things up a little and clear some space - maybe a few more will be migrated.
import numpy as np
LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

#if we want fast NUMBA factorials
@jit(nopython=True)
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]

#can be used for pk and bk
@jit(nopython=True)#counts the distinct permuations (n1+n2+..)!/n1! n2!...
def cdp_f123(f123): #f123 has done most of the computation for us and also now with NUMBA
    dem = 1
    for i in f123:
        dem *= fast_factorial(i)
    return fast_factorial(np.sum(f123))/dem

#no longer really used as they aren't necessary/cumbersome -8/10/22
#zero errors can be avoided more easily by setting the troublesome array 0 values to 1


#using divivison with np.where(b==0,0,a/b) is conditional selection not conditional execution so in order for no zero division errors
# particularly important for numba
def div_without_zero(a,b):
    mask = (b != 0)
    mask1 = (b == 0)
    out = a.copy()
    out[mask] /= b[mask]
    out[mask1] = 0
    return out

#as above just for division of scalars
def scalar_div(a,b):
    if b == 0:
        return 0
    else:
        return a/b
    
#used to compute distinct permutations as implied - used in Qpqrs() etc in expanding 
#probably numpy function or something but this is fine
def count_distinct_permutations(comb): #counts the distinct permuations (n1+n2+..)!/n1! n2!...
    if comb.size == 0:
        #print('count_distinct_permutations thinks this is the Monopole') #probably don't need this warning
        return 1
    x = np.bincount(comb) #get number of occurences per integer
    y = np.array(x[x != 0]) #get rid of integers that dont appear
    dem = 1
    for i in range(len(y)):
        dem *= np.math.factorial(y[i]) 
    return np.math.factorial(len(comb))/(dem)

#returns triangle numbers - which is how many terms there are for each multipole (or at least to whcih order)
def Num_terms(l):
    return np.int((l+1)*(l+2)/2)