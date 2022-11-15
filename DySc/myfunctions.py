import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import scipy
import math
from scipy import special
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.interpolate import griddata, interp2d
from scipy.special import ellipe,ellipk
au = 1.5e13
msun = 2e33
Gcgs = 6.67e-8
G = Gcgs * (msun/(au)) * 1e-10  #msun au cm/s

a = np.arange(1,1000,0.5)
b = np.arange(1,200,0.5)
a = a/100
b = b/100
I = np.loadtxt('Int1.txt')
f = interp2d(a,b, z=I, fill_value=None )


def v0 (mstar, ro):
    
    return (G * mstar )/ ro


def vd (mdisc, ro):
    
    return (G * mdisc )/(2 * np.pi * ro )


def vstar (x, y):
    
    return x**2 / (x**2 + y**2)**(3/2)


def vp (x, y, hrf, rf, ro, q):
    
    '''c1 =  1 + q + 3/2
    c2 = 3/2 - q
    delta = 2 * q - 1
    return - x**(-1) * (c1 * hrf**2 * x**(-1*delta) * (rf/ro)**(-1*delta) - c2 * (y/x)**2)'''
    
    c1 =  1 + q + 3/2
    c2 = 3/2 - q
    delta = 2 * q - 1
    hr2_rescaled = hrf**2 * (rf/ro)**(-1*delta)
    A1 = -c1 * x**(-1*delta) #radial term
    A2 = - x**(2-2*q) #exp tapering term
    k = y/x
    B = 2/(1+k**2)**(3/2)*(1+1.5*k**2-(1+k**2)**(3/2)-(3/2-0.5*q)*(1+k**2-(1+k**2)**(3/2))) #vertical term
    return x**(-1) * (hr2_rescaled * (A1+A2) + B)


def vpstar (x, y, hrf, rf, ro, q):
    
    c1 =  1 + q + 3/2
    delta = 2 * q - 1
    return x**-1 * (1 - c1 * hrf**2 * x**(-1*delta) * (rf/ro)**(-1*delta) - 2*q * (1 - 1/np.sqrt(1 + (y/x)**2)) - 
                    hrf**2 * (rf/ro)**(-1*delta) * x**(1-delta))


def vdisc (x, y):
    
    return f(x,y)


def vdisco (x, y):
    
    a = np.arange(1,1000,0.5)
    b = np.arange(1,200,0.5)
    a = a/100
    b = b/100
    I = np.loadtxt('Int1.txt')
    f = interp2d(a,b, z=I, fill_value=None )
    I = f(x, y)
    return I[0]



def curve_total_model (x, y, mstar, mdisc, ro, hrf, rf, q):
    
   # star = v0 (mstar, ro) * vstar (x, y)
    #press = v0 (mstar, ro) * vp (x, y, hrf, rf, ro, q) 
    star_press = v0 (mstar, ro) * vpstar (x, y, hrf, rf, ro, q) 
    disc = vd (mdisc, ro) * vdisco (x, y)
    return (star_press + disc) ** 0.5



# ____ Model only Star _____

def curve_total1 (x, y, mstar):
    
    star = v0 (mstar, 1) * vstar (x, y)
    return (star) **0.5


def log_likelihood1(p, r, z, v, dv):
    mstar = p
    
    llkh1 = -0.5 * ((v - curve_total1 (r/1, z/1, mstar))**2 /(dv**2)
                  + np.log(2*np.pi*dv**2))   
    return np.sum(llkh1)


def log_prior1(p):
    
    mstar = p
    if (mstar < 0) or (mstar > 5):
        return -np.inf
    else:
        return -np.log(1)
    
    
def log_posterior1(p, r, z, v, dv):
    
    lp1 = log_prior1(p)
    if np.isfinite(lp1):
        return lp1 + log_likelihood1(p, r, z, v, dv)
    else:
        return lp1
# _____________________________________________________________


# ____ Model Star + PG _____

def curve_total2 (x, y, mstar,  hrf, rf, ro, q):
    
    
    return (v0 (mstar, ro) * vpstar (x, y, hrf, rf, ro, q) ) **0.5


def log_likelihood2(p, r, z, v, dv, hrf, rf, q):
    mstar, ro = p
    
    llkh2 = -0.5 * ((v - curve_total2 (r/ro, z/ro, mstar, hrf, rf, ro, q))**2 /(dv**2)
                  + np.log(2*np.pi*dv**2))   
    return np.sum(llkh2)


def log_prior2(p):
    
    mstar, ro = p
    if (mstar < 0) or (mstar > 5) or (ro < 50.) or (ro > 500.):
        return -np.inf
    else:
        return -np.log(1)
    
    

def log_posterior2(p, r, z, v, dv, hrf, rf, q):
    
    lp2 = log_prior2(p)
    if np.isfinite(lp2):
        return lp2 + log_likelihood2(p, r, z, v, dv, hrf, rf, q)
    else:
        return lp2
# _____________________________________________________________


# ____ Model Star + PG + SG _____

def curve_total (x, y, mstar, mdisc, ro, hrf, rf, q):
    
    #star = v0 (mstar, ro) * vstar (x, y)
    #press = v0 (mstar, ro) * vp (x, y, hrf, rf, ro, q) 
    star_press = v0 (mstar, ro) * vpstar (x, y, hrf, rf, ro, q) 
    disc = vd (mdisc, ro) * vdisc (x, y)
    return (star_press + disc) **0.5


def log_likelihood(p, r, z, v, dv, hrf, rf, q):
    mstar, mdisc, ro = p
    
    llkh = -0.5 * ((v - curve_total (r/ro, z/ro, mstar, mdisc, ro, hrf, rf, q))**2 /(dv**2)
                  + np.log(2*np.pi*dv**2))   
    return np.sum(llkh)


def log_prior(p):
    
    mstar, mdisc, ro = p
    if (mstar < 0) or (mstar > 5) or (mdisc < 0) or (ro < 50) or (ro > 300):
        return -np.inf
    else:
        return -np.log(1)

    
def log_posterior(p, r, z, v, dv, hrf, rf, q):
    
    lp = log_prior(p)
    if np.isfinite(lp):
        return lp + log_likelihood(p, r, z, v, dv, hrf, rf, q)
    else:
        return lp
# _____________________________________________________________

# ____ Model Star + PG + SG _____

def curve_total4 (x, y, mstar, ro, hrf, rf, q):
    
    #star = v0 (mstar, ro) * vstar (x, y)
    #press = v0 (mstar, ro) * vp (x, y, hrf, rf, ro, q) 
    star_press = v0 (mstar, ro) * vpstar (x, y, hrf, rf, ro, q) 
    return (star_press) **0.5


def log_likelihood4(p, r, z, v, dv, hrf, rf, q):
    mstar, ro = p
    
    llkh = -0.5 * ((v - curve_total4 (r/ro, z/ro, mstar, ro, hrf, rf, q))**2 /(dv**2)
                  + np.log(2*np.pi*dv**2))   
    return np.sum(llkh)


def log_prior4(p):
    
    mstar, ro = p
    if (mstar < 0) or (mstar > 5) or (ro < 150) or (ro > 300):
        return -np.inf
    else:
        return -np.log(1)

    
def log_posterior4(p, r, z, v, dv, hrf, rf, q):
    
    lp = log_prior4(p)
    if np.isfinite(lp):
        return lp + log_likelihood4(p, r, z, v, dv, hrf, rf, q)
    else:
        return lp