from numba import jit
f=0.5222389117917002
#so need to define different chunks of the power spectrum...
@jit(nopython=True)
def mono00(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1):
    return Pks*b1**2 + 2*Pks*b1*f*ks**2/(3*kk**2) + Pks*f**2*ks**4/(5*kk**4) 
@jit(nopython=True)
def mono02(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1):
    return Pks*(2*b1*f*kk**2 + 2*f**2*ks**2)/kk**4
@jit(nopython=True)
def mono04(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1):
    return Pks*f**2/kk**4
#1st order ----------------
@jit(nopython=True)
def mono15(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -2*1j*f**2*nu**5*(2*t - 1)*(Pkd*kk - 4*Pks)/(d*kk**6)
@jit(nopython=True)
def mono13(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return 1j*f*nu**3*(2*t - 1)*(-2*Pkd*kk*(10*f*ks**2 + kk**2*(3*b1 - 3*f)) + Pks*(12*b1*kk**2 - 12*f*kk**2 + 80*f*ks**2))/(3*d*kk**6)
@jit(nopython=True)
def mono11(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return 1j*f*nu*(2*t - 1)*(6*Pkd*kk*(kk - ks)*(kk + ks)*(b1*kk**2 + f*ks**2) + Pks*(12*b1*kk**2*ks**2 - 12*f*kk**2*ks**2 + 24*f*ks**4))/(3*d*kk**6)

#extra 2nd order terms to monopole and quadrupole (no dfog)
@jit(nopython=True)
def mono20(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return f*(-8*Pks*(7*b1*kk**2*(t*(2*t - 2) + 1)*(15*kk**4 - 25*kk**2*ks**2 + 12*ks**4) + 6*f*(t*(3*t - 3) + 1)*(35*kk**4*ks**2 - 63*kk**2*ks**4 + 30*ks**6)) - 35*kk**4*(2*Pks + kk*(4*Pkd + Pkdd*kk))*(3*b1*kk**2*(t*(2*t - 2) + 1) + f*ks**2*(t*(6*t - 6) + 1)) + 28*kk**2*(2*Pks*(5*b1*kk**2*(3*kk**2 - 2*ks**2)*(t*(2*t - 2) + 1) + 3*f*(1 - 2*t)**2*(5*kk**2*ks**2 - 4*ks**4)) + kk*(Pkd*(5*b1*kk**2*(3*kk**2 + ks**2)*(t*(2*t - 2) + 1) - 3*f*(1 - 2*t)**2*(-5*kk**2*ks**2 + ks**4)) + Pkdd*kk*(5*b1*kk**2*ks**2*(t*(2*t - 2) + 1) + 3*f*ks**4*(1 - 2*t)**2))) + 7*kk**2*(t*(2*t - 2) + 1)*(Pks*(kk + ks)*(30*kk - 30*ks)*(b1*kk**2 + 2*f*ks**2) + kk*(2*Pkd + Pkdd*kk)*(5*b1*kk**2*ks**2 + 3*f*ks**4)) - 2*kk*(2*Pkd*(35*b1*kk**2*(5*kk**2*ks**2 - 3*ks**4)*(t*(2*t - 2) + 1) + 27*f*(7*kk**2*ks**4 - 5*ks**6)*(t*(3*t - 3) + 1)) + 6*Pkdd*kk*(7*b1*kk**2*ks**4*(t*(2*t - 2) + 1) + 5*f*ks**6*(t*(3*t - 3) + 1))))/(105*d**2*kk**8)
@jit(nopython=True)
def mono22(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return f*nu**2*(-8*Pks*(7*b1*kk**2*(-75*kk**2 + 120*ks**2)*(t*(2*t - 2) + 1) + 6*f*(t*(3*t - 3) + 1)*(105*kk**4 - 630*kk**2*ks**2 + 630*ks**4)) - 105*f*kk**4*(2*Pks + kk*(4*Pkd + Pkdd*kk))*(t*(6*t - 6) + 1) + 7*kk**2*(Pks*(-600*f*ks**2 + 30*kk**2*(-3*b1 + 6*f)) + kk*(2*Pkd + Pkdd*kk)*(15*b1*kk**2 + 30*f*ks**2))*(t*(2*t - 2) + 1) + 28*kk**2*(2*Pks*(-30*b1*kk**2*(t*(2*t - 2) + 1) + 3*f*(1 - 2*t)**2*(15*kk**2 - 40*ks**2)) + kk*(Pkd*(15*b1*kk**2*(t*(2*t - 2) + 1) - 3*f*(1 - 2*t)**2*(-15*kk**2 + 10*ks**2)) + Pkdd*kk*(15*b1*kk**2*(t*(2*t - 2) + 1) + 30*f*ks**2*(1 - 2*t)**2))) - 2*kk*(2*Pkd*(35*b1*kk**2*(15*kk**2 - 30*ks**2)*(t*(2*t - 2) + 1) + 27*f*(70*kk**2*ks**2 - 105*ks**4)*(t*(3*t - 3) + 1)) + 6*Pkdd*kk*(70*b1*kk**2*ks**2*(t*(2*t - 2) + 1) + 105*f*ks**4*(t*(3*t - 3) + 1))))/(105*d**2*kk**8)
@jit(nopython=True)
def mono24(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return f*nu**4*(-8*Pks*(420*b1*kk**2*(t*(2*t - 2) + 1) + 6*f*(-315*kk**2 + 1050*ks**2)*(t*(3*t - 3) + 1)) + 7*kk**2*(-300*Pks*f + 15*f*kk*(2*Pkd + Pkdd*kk))*(t*(2*t - 2) + 1) + 28*kk**2*(-120*Pks*f*(1 - 2*t)**2 + kk*(-15*Pkd*f*(1 - 2*t)**2 + 15*Pkdd*f*kk*(1 - 2*t)**2)) - 2*kk*(2*Pkd*(-525*b1*kk**2*(t*(2*t - 2) + 1) + 27*f*(35*kk**2 - 175*ks**2)*(t*(3*t - 3) + 1)) + 6*Pkdd*kk*(35*b1*kk**2*(t*(2*t - 2) + 1) + 175*f*ks**2*(t*(3*t - 3) + 1))))/(105*d**2*kk**8)
@jit(nopython=True)
def mono26(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return f*nu**6*(-10080*Pks*f*(t*(3*t - 3) + 1) - 2*kk*(-1890*Pkd*f*(t*(3*t - 3) + 1) + 210*Pkdd*f*kk*(t*(3*t - 3) + 1)))/(105*d**2*kk**8)


#dipole -------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def dipo01(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1):
    return 4*Pks*f*ks*(5*b1*kk**2 + 3*f*ks**2)/(5*kk**4)
@jit(nopython=True)
def dipo03(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1):
    return 4*Pks*f**2*ks*nu**3/kk**4
#1st order ----------------------------------
@jit(nopython=True)
def dipo14(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -10*1j*f**2*ks*nu**4*(2*t - 1)*(Pkd*kk - 4*Pks)/(d*kk**6)
@jit(nopython=True)
def dipo12(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return 6*1j*f*ks*nu**2*(2*t - 1)*(-Pkd*b1*kk**3 + Pkd*f*kk**3 - 2*Pkd*f*kk*ks**2 + 2*Pks*b1*kk**2 - 2*Pks*f*kk**2 + 8*Pks*f*ks**2)/(d*kk**6)
@jit(nopython=True)
def dipo10(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return 2*1j*f*ks*(2*t - 1)*(Pkd*kk*(35*b1*kk**4 - 21*b1*kk**2*ks**2 + 21*f*kk**2*ks**2 - 15*f*ks**4) + Pks*(60*f*ks**4 + kk**2*ks**2*(42*b1 - 42*f)))/(35*d*kk**6)
#2nd order --------------------------------
@jit(nopython=True)
def dipo25(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -2*f*ks*nu**5*(-3780*Pkd*f*kk*(3*t**2 - 3*t + 1) + 420*Pkdd*f*kk**2*(3*t**2 - 3*t + 1) + 10080*Pks*f*(3*t**2 - 3*t + 1))/(35*d**2*kk**8)
@jit(nopython=True)
def dipo23(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return - 2*f*ks*nu**3*(2*Pkd*kk*(-700*b1*kk**2*(2*t**2 - 2*t + 1) - 2*f*(-35*kk**2*(60*t**2 - 60*t + 19) + 1890*ks**2*(3*t**2 - 3*t + 1))) + Pkdd*kk**2*(280*b1*kk**2*(2*t**2 - 2*t + 1) + f*(-70*kk**2*(18*t**2 - 18*t + 5) + 840*ks**2*(3*t**2 - 3*t + 1))) + 2240*Pks*b1*kk**2*(2*t**2 - 2*t + 1) + 2*Pks*f*(-140*kk**2*(66*t**2 - 66*t + 23) + 10080*ks**2*(3*t**2 - 3*t + 1)))/(35*d**2*kk**8)
@jit(nopython=True)
def dipo21(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return - 2*f*ks*nu*(2*Pkd*kk*(35*b1*kk**2*(7*kk**2 - 12*ks**2)*(2*t**2 - 2*t + 1) - 2*f*(70*kk**4*(3*t**2 - 3*t + 1) - 21*kk**2*ks**2*(60*t**2 - 60*t + 19) + 405*ks**4*(3*t**2 - 3*t + 1))) + Pkdd*kk**2*(-7*b1*kk**2*(25*kk**2 - 24*ks**2)*(2*t**2 - 2*t + 1) + f*(35*kk**4*(6*t**2 - 6*t + 1) - 42*kk**2*ks**2*(18*t**2 - 18*t + 5) + 180*ks**4*(3*t**2 - 3*t + 1))) - 14*Pks*b1*kk**2*(45*kk**2 - 96*ks**2)*(2*t**2 - 2*t + 1) + 2*Pks*f*(35*kk**4*(18*t**2 - 18*t + 7) - 84*kk**2*ks**2*(66*t**2 - 66*t + 23) + 2160*ks**4*(3*t**2 - 3*t + 1)))/(35*d**2*kk**8)
#third order -------------------------------
@jit(nopython=True)
def dipo30(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return 2*1j*f*ks*(2*t - 1)*(-24*Pks*(3*b1*kk**2*(35*kk**4 - 154*kk**2*ks**2 + 120*ks**4)*(t**2 - t + 1) + f*(-35*kk**6*(3*t**2 - 3*t + 2) + 168*kk**4*ks**2*(9*t**2 - 9*t + 5) - 135*kk**2*ks**4*(27*t**2 - 27*t + 14) + 1120*ks**6*(2*t**2 - 2*t + 1))) + kk*(6*Pkd*(2*b1*kk**2*(175*kk**4 - 672*kk**2*ks**2 + 495*ks**4)*(t**2 - t + 1) + f*(-35*kk**6*(6*t**2 - 6*t + 5) + 21*kk**4*ks**2*(162*t**2 - 162*t + 79) - 135*kk**2*ks**4*(53*t**2 - 53*t + 26) + 2030*ks**6*(2*t**2 - 2*t + 1))) + kk*(-6*Pkdd*(10*b1*kk**2*(14*kk**4 - 42*kk**2*ks**2 + 27*ks**4)*(t**2 - t + 1) + f*(-35*kk**6*(6*t**2 - 6*t + 1) + 21*kk**4*ks**2*(38*t**2 - 38*t + 15) - 45*kk**2*ks**4*(31*t**2 - 31*t + 14) + 350*ks**6*(2*t**2 - 2*t + 1))) + Pkddd*kk*(6*b1*kk**2*(35*kk**4 - 63*kk**2*ks**2 + 30*ks**4)*(t**2 - t + 1) + f*(126*kk**4*ks**2*(1 - 2*t)**2 - 135*kk**2*ks**4*(5*t**2 - 5*t + 2) + 140*ks**6*(2*t**2 - 2*t + 1))))))/(105*d**3*kk**10)
@jit(nopython=True)
def dipo32(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return 2*1j*f*ks*nu**2*(2*t - 1)*(-24*Pks*(3*b1*kk**2*(-770*kk**2 + 1680*ks**2)*(t**2 - t + 1) + f*(840*kk**4*(9*t**2 - 9*t + 5) - 1890*kk**2*ks**2*(27*t**2 - 27*t + 14) + 30240*ks**4*(2*t**2 - 2*t + 1))) + kk*(6*Pkd*(2*b1*kk**2*(-3360*kk**2 + 6930*ks**2)*(t**2 - t + 1) + f*(105*kk**4*(162*t**2 - 162*t + 79) - 1890*kk**2*ks**2*(53*t**2 - 53*t + 26) + 54810*ks**4*(2*t**2 - 2*t + 1))) + kk*(-6*Pkdd*(10*b1*kk**2*(-210*kk**2 + 378*ks**2)*(t**2 - t + 1) + f*(105*kk**4*(38*t**2 - 38*t + 15) - 630*kk**2*ks**2*(31*t**2 - 31*t + 14) + 9450*ks**4*(2*t**2 - 2*t + 1))) + Pkddd*kk*(6*b1*kk**2*(-315*kk**2 + 420*ks**2)*(t**2 - t + 1) + f*(630*kk**4*(1 - 2*t)**2 - 1890*kk**2*ks**2*(5*t**2 - 5*t + 2) + 3780*ks**4*(2*t**2 - 2*t + 1))))))/(105*d**3*kk**10)
@jit(nopython=True)
def dipo34(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return 2*1j*f*ks*nu**4*(2*t - 1)*(-24*Pks*(4200*b1*kk**2*(t**2 - t + 1) + f*(-1575*kk**2*(27*t**2 - 27*t + 14) + 70560*ks**2*(2*t**2 - 2*t + 1))) + kk*(6*Pkd*(11550*b1*kk**2*(t**2 - t + 1) + f*(-1575*kk**2*(53*t**2 - 53*t + 26) + 127890*ks**2*(2*t**2 - 2*t + 1))) + kk*(-6*Pkdd*(3150*b1*kk**2*(t**2 - t + 1) + f*(-525*kk**2*(31*t**2 - 31*t + 14) + 22050*ks**2*(2*t**2 - 2*t + 1))) + Pkddd*kk*(2100*b1*kk**2*(t**2 - t + 1) + f*(-1575*kk**2*(5*t**2 - 5*t + 2) + 8820*ks**2*(2*t**2 - 2*t + 1))))))/(105*d**3*kk**10)
@jit(nopython=True)
def dipo36(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return 2*1j*f*ks*nu**6*(2*t - 1)*(-564480*Pks*f*(2*t**2 - 2*t + 1) + kk*(255780*Pkd*f*(2*t**2 - 2*t + 1) + kk*(-44100*Pkdd*f*(2*t**2 - 2*t + 1) + 2940*Pkddd*f*kk*(2*t**2 - 2*t + 1))))/(105*d**3*kk**10)

#quadrupole -------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def quad00(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1):
    return 4*Pks*f*ks**2*(7*b1*kk**2 + 3*f*ks**2)/(21*kk**4)
@jit(nopython=True)
def quad02(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1):
    return 4*Pks*f**2*ks**2/kk**4
#1st order -------------------------------------
@jit(nopython=True)
def quad13(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -40*1j*f**2*ks**2*nu**3*(2*t - 1)*(Pkd*kk - 4*Pks)/(3*d*kk**6)
@jit(nopython=True)
def quad11(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return 4*1j*f*ks**2*nu*(2*t - 1)*(-7*Pkd*b1*kk**3 + 7*Pkd*f*kk**3 - 10*Pkd*f*kk*ks**2 + 14*Pks*b1*kk**2 - 14*Pks*f*kk**2 + 40*Pks*f*ks**2)/(7*d*kk**6)
#2nd order ------------------------------------------ 
@jit(nopython=True)
def quad20(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -2*f*ks**2*(2*Pks*(-3*b1*kk**2*(21*kk**2 - 32*ks**2)*(2*t**2 - 2*t + 1) + f*(7*kk**4*(18*t**2 - 18*t + 7) - 12*kk**2*ks**2*(66*t**2 - 66*t + 23) + 240*ks**4*(3*t**2 - 3*t + 1))) + kk*(2*Pkd*(b1*kk**2*(49*kk**2 - 60*ks**2)*(2*t**2 - 2*t + 1) - 2*f*(14*kk**4*(3*t**2 - 3*t + 1) - 3*kk**2*ks**2*(60*t**2 - 60*t + 19) + 45*ks**4*(3*t**2 - 3*t + 1))) + Pkdd*kk*(b1*kk**2*(-35*kk**2 + 24*ks**2)*(2*t**2 - 2*t + 1) + f*(7*kk**4*(6*t**2 - 6*t + 1) - 6*kk**2*ks**2*(18*t**2 - 18*t + 5) + 20*ks**4*(3*t**2 - 3*t + 1)))))/(21*d**2*kk**8)
@jit(nopython=True)
def quad22(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -2*f*ks**2*nu**2*(2*Pks*(672*b1*kk**2*(2*t**2 - 2*t + 1) + f*(-84*kk**2*(66*t**2 - 66*t + 23) + 4320*ks**2*(3*t**2 - 3*t + 1))) + kk*(2*Pkd*(-420*b1*kk**2*(2*t**2 - 2*t + 1) - 2*f*(-21*kk**2*(60*t**2 - 60*t + 19) + 810*ks**2*(3*t**2 - 3*t + 1))) + Pkdd*kk*(168*b1*kk**2*(2*t**2 - 2*t + 1) + f*(-42*kk**2*(18*t**2 - 18*t + 5) + 360*ks**2*(3*t**2 - 3*t + 1)))))/(21*d**2*kk**8)
@jit(nopython=True)
def quad24(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -2*f*ks**2*nu**4*(10080*Pks*f*(3*t**2 - 3*t + 1) + kk*(-3780*Pkd*f*(3*t**2 - 3*t + 1) + 420*Pkdd*f*kk*(3*t**2 - 3*t + 1)))/(21*d**2*kk**8)

#octopole --------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def octo01(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1):
    return 8*Pks*f**2*ks**3/(5*kk**4)
#1st order -------------------------------------
@jit(nopython=True)
def octo12(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -8*1j*f**2*ks**3*nu**2*(2*t - 1)*(Pkd*kk - 4*Pks)/(d*kk**6)
@jit(nopython=True)
def octo10(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return - 4*1j*f*ks**3*(2*t - 1)*(9*Pkd*b1*kk**3 - 9*Pkd*f*kk**3 + 10*Pkd*f*kk*ks**2 - 18*Pks*b1*kk**2 + 18*Pks*f*kk**2 - 40*Pks*f*ks**2)/(45*d*kk**6)
#second ------------------------------------------
@jit(nopython=True)
def octo21(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -8*f*ks**3*nu*(12*Pks*(8*b1*kk**2*(t*(2*t - 2) + 1) + f*kk**2*(-t*(66*t - 66) - 23) + 40*f*ks**2*(t*(3*t - 3) + 1)) + kk*(-6*Pkd*(10*b1*kk**2*(t*(2*t - 2) + 1) + f*kk**2*(-t*(60*t - 60) - 19) + 30*f*ks**2*(t*(3*t - 3) + 1)) + Pkdd*kk*(12*b1*kk**2*(t*(2*t - 2) + 1) - 3*f*kk**2*(t*(18*t - 18) + 5) + 20*f*ks**2*(t*(3*t - 3) + 1))))/(15*d**2*kk**8)
@jit(nopython=True)
def octo23(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -8*f*ks**3*nu**3*(1440*Pks*f*(t*(3*t - 3) + 1) + kk*(-540*Pkd*f*(t*(3*t - 3) + 1) + 60*Pkdd*f*kk*(t*(3*t - 3) + 1)))/(15*d**2*kk**8)
#third --------------------------------------------
@jit(nopython=True)
def octo30(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -8*1j*f*ks**3*(2*t - 1)*(3*Pkd*kk*(44*b1*kk**2*(48*kk**2 - 55*ks**2)*(t**2 - t + 1) - 3*f*(11*kk**4*(162*t**2 - 162*t + 79) - 110*kk**2*ks**2*(53*t**2 - 53*t + 26) + 2030*ks**4*(2*t**2 - 2*t + 1))) - 264*Pks*b1*kk**2*(33*kk**2 - 40*ks**2)*(t**2 - t + 1) + 72*Pks*f*(44*kk**4*(9*t**2 - 9*t + 5) - 55*kk**2*ks**2*(27*t**2 - 27*t + 14) + 560*ks**4*(2*t**2 - 2*t + 1)) + kk**2*(-3*Pkdd*(660*b1*kk**2*(kk**2 - ks**2)*(t**2 - t + 1) - 33*f*kk**4*(38*t**2 - 38*t + 15) + 110*f*kk**2*ks**2*(31*t**2 - 31*t + 14) - 1050*f*ks**4*(2*t**2 - 2*t + 1)) + Pkddd*kk*(11*b1*kk**2*(27*kk**2 - 20*ks**2)*(t**2 - t + 1) - 3*f*(33*kk**4*(1 - 2*t)**2 - 55*kk**2*ks**2*(5*t**2 - 5*t + 2) + 70*ks**4*(2*t**2 - 2*t + 1)))))/(495*d**3*kk**10)
@jit(nopython=True)
def octo32(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -8*1j*f*ks**3*nu**2*(2*t - 1)*(3*Pkd*kk*(-21780*b1*kk**2*(t**2 - t + 1) - 3*f*(-990*kk**2*(53*t**2 - 53*t + 26) + 44660*ks**2*(2*t**2 - 2*t + 1))) + 95040*Pks*b1*kk**2*(t**2 - t + 1) + 72*Pks*f*(-495*kk**2*(27*t**2 - 27*t + 14) + 12320*ks**2*(2*t**2 - 2*t + 1)) + kk**2*(-3*Pkdd*(-5940*b1*kk**2*(t**2 - t + 1) + 990*f*kk**2*(31*t**2 - 31*t + 14) - 23100*f*ks**2*(2*t**2 - 2*t + 1)) + Pkddd*kk*(-1980*b1*kk**2*(t**2 - t + 1) - 3*f*(-495*kk**2*(5*t**2 - 5*t + 2) + 1540*ks**2*(2*t**2 - 2*t + 1)))))/(495*d**3*kk**10)
@jit(nopython=True)
def octo34(Pks,Pkd,Pkdd,Pkddd,kk,ks,t,b1=1,nu=1,d=1):
    return -8*1j*f*ks**3*nu**4*(2*t - 1)*(-602910*Pkd*f*kk*(2*t**2 - 2*t + 1) + 1330560*Pks*f*(2*t**2 - 2*t + 1) + kk**2*(103950*Pkdd*f*(2*t**2 - 2*t + 1) - 6930*Pkddd*f*kk*(2*t**2 - 2*t + 1)))/(495*d**3*kk**10)

