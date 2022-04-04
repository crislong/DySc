class rotationcurve:
    
    def __init__(self, v, dv, r, z, hrt, rt, q, d = None, model = 3, 
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
            
            if (err = True):
                ax.errorbar(self.radius, self.velocity, yerr = self.errvelocity, c='r', lw=1.0, capsize=2.0,
                            capthick=1.0, fmt=' ', label='Data')
                
        if (self.model == 2):
            ax.plot(self.radius, curve_total2 (self.radius / self.bestfitro,
                self.elayer / self.bestfitro, self.bestfitmstar, 
                self.aspectratio, self.typicalradius, , c='black', lw=1, label='Model')
            ax.plot(self.radius, self.velocity, '.', c='orange')
                    
            if (err = True):
                ax.errorbar(self.radius, self.velocity, yerr = self.errvelocity, c='r', lw=1.0, capsize=2.0,
                            capthick=1.0, fmt=' ', label='Data')
                    
                    
        if (self.model == 1):
            ax.plot(self.radius, curve_total1 (self.radius / self.bestfitro,
                self.elayer / self.bestfitro, self.bestfitmstar, c='black', lw=1, label='Model')
            ax.plot(self.radius, self.velocity, '.', c='orange')
                    
            if (err = True):
                ax.errorbar(self.radius, self.velocity, yerr = self.errvelocity, c='r', lw=1.0, capsize=2.0,
                            capthick=1.0, fmt=' ', label='Data')
                    
        ax.legend()
        ax.set_xlabel(r'Radius [au]', size=18)
        ax.set_ylabel(r'$v_\phi (R,z)$ [km/s]', size=18)
            
            
