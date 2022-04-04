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


def vp (x, y, hrf, rf, ro):
    
    c1 =  1 + 0.25 + 3/2
    c2 = 3/2 - 0.25
    delta = 2 * 0.25 - 1
    return - x**(-1) * (c1 * hrf**2 * x**(-1*delta) * (rf/ro)**(-1*delta) - c2 * (y/x)**2)


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



def curve_total_model (x, y, mstar, mdisc, ro, hrf, rf):
    
    star = v0 (mstar, ro) * vstar (x, y)
    press = v0 (mstar, ro) * vp (x, y, hrf, rf, ro) 
    disc = vd (mdisc, ro) * vdisco (x, y)
    return (star + press + disc) ** 0.5



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

def curve_total2 (x, y, mstar, hrf, rf):
    
    star = v0 (mstar, rf) * vstar (x, y)
    press = v0 (mstar, rf) * vp (x, y, hrf, rf, rf) 
    return (star + press) **0.5


def log_likelihood2(p, r, z, v, dv, hrf, rf):
    mstar = p
    
    llkh2 = -0.5 * ((v - curve_total2 (r/rf, z/rf, mstar, hrf, rf))**2 /(dv**2)
                  + np.log(2*np.pi*dv**2))   
    return np.sum(llkh2)


def log_prior2(p):
    
    mstar = p
    if (mstar < 0) or (mstar > 5):
        return -np.inf
    else:
        return -np.log(1)
    
    

def log_posterior2(p, r, z, v, dv, hrf, rf):
    
    lp2 = log_prior2(p)
    if np.isfinite(lp2):
        return lp2 + log_likelihood2(p, r, z, v, dv, hrf, rf)
    else:
        return lp2
# _____________________________________________________________


# ____ Model Star + PG + SG _____

def curve_total (x, y, mstar, mdisc, ro, hrf, rf):
    
    star = v0 (mstar, ro) * vstar (x, y)
    press = v0 (mstar, ro) * vp (x, y, hrf, rf, ro) 
    disc = vd (mdisc, ro) * vdisc (x, y)
    return (star + press + disc) **0.5


def log_likelihood(p, r, z, v, dv, hrf, rf):
    mstar, mdisc, ro = p
    
    llkh = -0.5 * ((v - curve_total (r/ro, z/ro, mstar, mdisc, ro, hrf, rf))**2 /(dv**2)
                  + np.log(2*np.pi*dv**2))   
    return np.sum(llkh)


def log_prior(p):
    
    mstar, mdisc, ro = p
    if (mstar < 0) or (mstar > 5) or (mdisc < 0) or (ro < 50) or (ro > 1000):
        return -np.inf
    else:
        return -np.log(1)

    
def log_posterior(p, r, z, v, dv, hrf, rf):
    
    lp = log_prior(p)
    if np.isfinite(lp):
        return lp + log_likelihood(p, r, z, v, dv, hrf, rf)
    else:
        return lp
# _____________________________________________________________



class rotationcurve:
    
    def __init__(self, v, dv, r, z, hrt, rt, q = 0.25, d = None, model = 3, 
                 fro = True, nt = 1, simultaneous = True):
        
        ''' Class rotation curve
        v = input velocity vector [km/s]
        dv = input velocity error vector [km/s]
        r = input radial vector [au]
        z = input height vector [au]
        hrt = H/R @ r = rt
        rt = typical radius for HR [au]
        q = sound speed power law coefficient 
        d = distance of the source [pc], not essential
        SG = self gravitating fit (default True)
        outerfit = outer radius fit (delfault True)'''
        
        self.velocity = v 
        self.errvelocity = dv
        self.radius = r
        self.elayer = z
        self.aspectratio = hrt
        self.typicalradius = rt
        self.plcoefficient = q
        self.distance = d
        self.model = model
        self.outfit = fro
        self.ntracers = nt
        self.simultaneousfit = simultaneous
        
        self.bestfitmstar = 0.
        self.bestfitmdisc = 0.
        self.bestfitro = 0.
        
        if (self.ntracers > 1):
            if (self.velocity.shape[0] != nt or self.errvelocity.shape[0] != nt or
               self.radius.shape[0] != nt or self.radius.shape[0] != nt or
               self.aspectratio.shape != nt or self.typicalradius != nr or
               self.plcoefficient != nt):
                print('Error: Size of input data does not match with the numer of tracers')
                
        if (self.model !=1 and self.model !=2 and self.model !=3):
            print('Error: Wrong number of model. Valid numbers are: 1 Star, 2 Star + PG, 3 Star + PG + SG')
        
        
    def fit(self, nw, nb, ns, Qcorner = True):
        
      #  a = np.arange(1,1000,0.5)
       # b = np.arange(1,200,0.5)
        #a = a/100
   #     b = b/100
    #    I = np.loadtxt('Int1.txt')
     #   f = interp2d(a,b, z=I, fill_value=None )
        
        nwalkers = nw
        nburnin = nb
        nsteps = ns
        
        if (self.model == 3):
            ndim = 3
            pos = np.random.uniform(low=[0.5, 0.05, 100], high=[1.5, 0.15, 400], size=(nwalkers, ndim))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                args=(self.radius, self.elayer, self.velocity, self.errvelocity,
                          self.aspectratio, self.typicalradius) )
            state = sampler.run_mcmc(pos, nburnin)
            sampler.reset();
            sampler.run_mcmc(state, nsteps, progress=True)
            if (Qcorner == True):
                corner.corner(sampler.flatchain, bins=100, quantiles=[0.025, 0.5, 0.975], show_titles=True, 
                titles=[r"$M_{\star}$",r"$M_{\rm disc}$",r"$R_{\rm c}$"], 
                labels=[r"$M_{\star}$",r"$M_{\rm disc}$",r"$R_{\rm c}$"], 
                title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 15}, title_fmt='.5f');
                
            self.bestfitmstar = np.quantile(sampler.flatchain[:,0], 0.5)
            self.bestfitmdisc = np.quantile(sampler.flatchain[:,1], 0.5)
            self.bestfitro = np.quantile(sampler.flatchain[:,2], 0.5)
            
        if (self.model == 2):
            ndim = 1
            pos = np.random.uniform(low=[0.5], high=[1.5], size=(nwalkers, ndim))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior2, 
                args=(self.radius, self.elayer, self.velocity, self.errvelocity,
                          self.aspectratio, self.typicalradius) )
            state = sampler.run_mcmc(pos, nburnin)
            sampler.reset();
            sampler.run_mcmc(state, nsteps, progress=True)
            if (Qcorner == True):
                corner.corner(sampler.flatchain, bins=100, quantiles=[0.025, 0.5, 0.975], show_titles=True, 
                titles=[r"$M_{\star}$"], 
                labels=[r"$M_{\star}$"], 
                title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 15}, title_fmt='.5f');
            
            self.bestfitmstar = np.quantile(sampler.flatchain[:], 0.5)
                
                
        if (self.model == 1):
            ndim = 1
            pos = np.random.uniform(low=[0.5], high=[1.5], size=(nwalkers, ndim))
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior1, 
                args=(self.radius, self.elayer, self.velocity, self.errvelocity) )
            state = sampler.run_mcmc(pos, nburnin)
            sampler.reset();
            sampler.run_mcmc(state, nsteps, progress=True)
            if (Qcorner == True):
                corner.corner(sampler.flatchain, bins=100, quantiles=[0.025, 0.5, 0.975], show_titles=True, 
                titles=[r"$M_{\star}$"], 
                labels=[r"$M_{\star}$"], 
                title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 15}, title_fmt='.5f');
                
            self.bestfitmstar = np.quantile(sampler.flatchain[:], 0.5)
            
            
    def plot_model(self, err = True):
        
        fig, (ax) = plt.subplots(1,1,figsize=(8,8))
    
        if (self.model == 3):
            ax.plot(self.radius, curve_total_model(self.radius / self.bestfitro,
                self.elayer / self.bestfitro, self.bestfitmstar, self.bestfitmdisc, self.bestfitro,
                self.aspectratio, self.typicalradius), c='black', lw=1, label='Model')
            ax.plot(self.radius, self.velocity, '.', c='orange')
            if (err == True):
                ax.errorbar(self.radius, self.velocity, yerr = self.errvelocity, c='r', lw=1.0, capsize=2.0,
                 capthick=1.0, fmt=' ', label='Data')
                
        if (self.model == 2):
            ax.plot(self.radius, curve_total2(self.radius / self.typicalradius,
                self.elayer / self.typicalradius, self.bestfitmstar,
                self.aspectratio, self.typicalradius), c='black', lw=1, label='Model')
            ax.plot(self.radius, self.velocity, '.', c='orange')
            if (err == True):
                ax.errorbar(self.radius, self.velocity, yerr = self.errvelocity, c='r', lw=1.0, capsize=2.0,
                 capthick=1.0, fmt=' ', label='Data')
                
        if (self.model == 1):
            ax.plot(self.radius, curve_total1(self.radius,
                self.elayer, self.bestfitmstar), c='black', lw=1, label='Model')
            ax.plot(self.radius, self.velocity, '.', c='orange')
            if (err == True):
                ax.errorbar(self.radius, self.velocity, yerr = self.errvelocity, c='r', lw=1.0, capsize=2.0,
                 capthick=1.0, fmt=' ', label='Data')
            
        ax.legend()
        ax.set_xlabel(r'Radius [au]', size=18)
        ax.set_ylabel(r'$v_\phi (R,z)$ [km/s]', size=18)