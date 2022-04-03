#
# ALL FUNCTIONS
# Cristiano Longarini
#

import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import scipy
import math
from scipy import special
from scipy.interpolate import griddata, interp2d
from scipy.special import ellipe,ellipk

# constants 
au = 1.5e13
msun = 2e33
Gcgs = 6.67e-8
G = Gcgs * (msun/(au)) * 1e-10 



def v0 (mstar, ro):

    ''' Dimensionality constant for Star and Pressure contribution
    mstar = mass of the central star [msun]
    ro = outer radius of the disc [au] '''
    
    return (G * mstar )/ ro


def vd (mdisc, ro):

    ''' Dimensionality constant for Disc contribution
    mdisc = mass of the disc [msun]
    ro = outer radius of the disc [au] '''
    
    return (G * mdisc )/(2 * np.pi * ro )


def vstar (x, y):

    ''' Dimensionless Star contribution to the velocity
    x = dimensionless radius
    y = dimensionless height '''
    
    return x**2 / (x**2 + y**2)**(3/2)


def vp (x, y, hrf, rf, ro):

    ''' Dimensionless Pressure contribution to the velocity
    x = dimensionless radius
    y = dimensionless height
    hrf = H/R @ r = rf
    rf = typical radius for H/R [au] '''
    
    c1 =  1 + 0.25 + 3/2
    c2 = 3/2 - 0.25
    delta = 2 * 0.25 - 1
    return - x**(-1) * (c1 * hrf**2 * x**(-1*delta) * (rf/ro)**(-1*delta) - c2 * (y/x)**2)


def vdisc (x, y):

    ''' Dimensionless Disc contribution to the velocity
    x = dimensionless radius
    y = dimensionless height '''
    
    return f(x,y)


def vdisco (x, y):

    ''' Dimensionless Star contribution to the velocity (for the model)
    x = dimensionless radius
    y = dimensionless height '''
    
    a = np.arange(1,1000,0.5)
    b = np.arange(1,200,0.5)
    a = a/100
    b = b/100
    I = np.loadtxt('Int1.txt')
    f = interp2d(a,b, z=I, fill_value=None )
    I = f(x, y)
    return I[0]


def curve_total (x, y, mstar, mdisc, ro, hrf, rf):

    ''' Rotation curve
    x = dimensionless radius
    y = dimensionless height 
    mstar = mass of the star [msun] 
    mdisc = mass of the disc [msun]
    ro = outer radius [au]
    hrf = H/R # r = rf
    rf = typical radius of H/R [au] '''
    
    star = v0 (mstar, ro) * vstar (x, y)
    press = v0 (mstar, ro) * vp (x, y, hrf, rf, ro) 
    disc = vd (mdisc, ro) * vdisc (x, y)
    return (star + press + disc) **0.5


def curve_total_model (x, y, mstar, mdisc, ro, hrf, rf):
    
    ''' Rotation curve (model)
    x = dimensionless radius
    y = dimensionless height 
    mstar = mass of the star [msun] 
    mdisc = mass of the disc [msun]
    ro = outer radius [au]
    hrf = H/R # r = rf
    rf = typical radius of H/R [au] '''
    
    star = v0 (mstar, ro) * vstar (x, y)
    press = v0 (mstar, ro) * vp (x, y, hrf, rf, ro) 
    disc = vd (mdisc, ro) * vdisco (x, y)
    return (star + press + disc) ** 0.5


def log_likelihood(p, r, z, v, dv, hrf, rf):

    ''' Log-Likelihood of the complete model
    p = vector of best fit parameters
    r = radial input vector [au]
    z = vertical input vector [au]
    v = velocity input vector [km/s]
    dv = error velocity input vector [km/s]
    hrf = H/R # r = rf
    rf = typical radius of H/R [au] '''

    mstar, mdisc, ro = p
    
    llkh = -0.5 * ((v - curve_total (r/ro, z/ro, mstar, mdisc, ro, hrf, rf))**2 /(dv**2)
                  + np.log(2*np.pi*dv**2))   
    return np.sum(llkh)


def log_prior(p):

    ''' Log-Priors of the complete model
    p = vector of best fit parameters '''
    
    mstar, mdisc, ro = p
    
    if (mstar < 0) or (mstar > 2) or (mdisc < 0) or (ro < 50) or (ro > 1000):
        return -np.inf
    else:
        return -np.log(1)

    
def log_posterior(p, r, z, v, dv, hrf, rf):

    ''' Log-Posterior of the complete model
    p = vector of best fit parameters
    r = radial input vector [au]
    z = vertical input vector [au]
    v = velocity input vector [km/s]
    dv = error velocity input vector [km/s]
    hrf = H/R # r = rf
    rf = typical radius of H/R [au] '''
    
    lp = log_prior(p)
    if np.isfinite(lp):
        return lp + log_likelihood(p, r, z, v, dv, hrf, rf)
    else:
        return lp