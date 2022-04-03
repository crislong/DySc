class rotationcurve:
    
    def __init__(self, v, dv, r, z, hrt, rt, q, d = None, SG = True, outerfit = True):
        
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
        self.selfgr = SG
        self.outfit = outerfit
        
    def fit(self, nwalkers, nburnin, nsteps, Qcorner = True):
        
        nwalkers = 250
        ndim = 3
        nburnin = 1000
        nsteps = 750
            
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
