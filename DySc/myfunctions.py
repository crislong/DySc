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
from functools import partial
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





#____ Thermal stratification routines _____

def keplerian2(mstar, radius):
    
    return Gcgs * msun * mstar / (radius * au)


def keplerian_height2(mstar, radius, height):
    
    return Gcgs * msun * mstar * au**2 * radius**2 / ((radius**2 * au**2 + height**2 * au**2)**(3/2))


def hr(hr100, radius, q):
    
    return hr100 * (radius/100)**(-q/2 + 0.5)

def temperature(radius, q_t, t100):
    
    return t100 * (radius/100)**(-q_t)


def zq(radius, z0, beta):
    
    return z0 * (radius/100)**beta


def t_dullemond(radius, z, tmid100, q_mid, tatm100, q_atm, z0, beta, alpha):
    
    b = temperature(radius, q_atm, tatm100) / temperature(radius, q_mid, tmid100)
    z_q = zq(radius, z0, beta)
    return temperature(radius, q_mid, tmid100) * (1 + 0.5 * b**4 * (1+np.tanh(z/z_q - alpha)))**0.25


def integrand_fg_strat(zp, radius, tmid100, q_mid, tatm100, q_atm, z0, beta, alpha):
    
    return zp / t_dullemond(radius, zp, tmid100, q_mid, tatm100, q_atm, z0, beta, alpha) * (1 + (zp/radius)**2 )**(-1.5)


def ln_fg_strat(radius, z_vector, hr100, tmid100, q_mid, tatm100, q_atm, z0, beta, alpha):
    
    h02 = - (radius * hr(hr100, radius, q_mid))**2
    grid_out = np.zeros([len(radius), len(z_vector)])
    grid_coordinatesR = np.zeros([len(radius), len(z_vector)])
    grid_coordinatesZ = np.zeros([len(radius), len(z_vector)])
    
    for i in range(len(radius)):
        for j in range(len(z_vector)):
            integrand_z = partial(integrand_fg_strat, radius = radius[i], tmid100 = tmid100, q_mid = q_mid, tatm100 = tatm100,
                                 q_atm = q_atm, z0 = z0, beta = beta, alpha = alpha)
            grid_out[i,j] = quad(integrand_z, 0, z_vector[j])[0] / h02[i]
            grid_coordinatesR[i,j] = radius[i]
            grid_coordinatesZ[i,j] = z_vector[j]

    return (grid_out, grid_coordinatesR, grid_coordinatesZ)


def write_stratification_files (radius, elayer, tmid100, q_mid, tatm100, q_atm, z0, beta, alpha, hr100, num_mol):

    f = t_dullemond(radius, elayer, tmid100, q_mid, tatm100, q_atm, z0, beta, alpha) / temperature(radius, q_mid, tmid100)
    a = ln_fg_strat(radius, np.linspace(0,radius[-1]/2,250), hr100, tmid100, q_mid, tatm100, q_atm, z0, beta, alpha)
    derivative = np.gradient(a[0], axis=0)
    contribution = griddata((a[1].flatten(), a[2].flatten()), derivative.flatten(), (radius,elayer)) * radius

    file1 = open('f_' + num_mol + '.txt', 'w')
    file2 = open('contrib_' + num_mol + '.txt', 'w')

    for i in range(len(f)):

        file1.write(str(f[i]) + '\n')
        file2.write(str(contribution[i]) + '\n')

    file1.close()
    file2.close()





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

def log_posterior1_sim(p, r, z, v, dv, r2, z2, v2, dv2):

    lp1 = log_prior1(p)
    if np.isfinite(lp1):
        return lp1 + log_likelihood1(p, r, z, v, dv) + log_likelihood1(p, r2, z2, v2, dv2)
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


def log_posterior2_sim(p, r, z, v, dv, hrf, rf, q, r2, z2, v2, dv2):
    
    lp2 = log_prior2(p)
    if np.isfinite(lp2):
        return lp2 + log_likelihood2(p, r, z, v, dv, hrf, rf, q) + log_likelihood2(p, r2, z2, v2, dv2, hrf, rf, q)
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


def log_posterior_sim(p, r, z, v, dv, hrf, rf, q, r2, z2, v2, dv2):
    
    lp = log_prior(p)
    if np.isfinite(lp):
        return lp + log_likelihood(p, r, z, v, dv, hrf, rf, q) + log_likelihood(p, r2, z2, v2, dv2, hrf, rf, q)
    else:
        return lp
# _____________________________________________________________