import numpy as np
from numpy import diag, exp, log, outer
import scipy.integrate as sint
import scipy.special as special

from structured import common, hmm, kalman
from structured.common import gemm, mult_diag



def lrep(x, n):
    """Replicate x n-times on a new first dimension"""
    shp = [1]
    shp.extend(x.shape)
    return x.reshape(shp).repeat(n, axis=0)


class _IntFun(object):
    def __init__(self, fun, eta, a,b,c, Z):
        # fun(lbd, sgm)
        self.fun = fun
        self.eta = eta
        self.a, self.b, self.c = a,b,c
        self.Z = Z

    def __call__(self, lbd):
        sgm = 1-lbd**2.
        exponent = -self.eta*log(sgm) + \
                   (self.a+lbd*self.b+lbd**2.*self.c)/sgm
        return self.fun(lbd,sgm)* exp(exponent)/self.Z


class Parameters(object):
    def __init__(self, dy, dc, dsc,
                 pi1, tT, p00, p11,
                 tlbd, lbds,
                 tnoise, noise_prec,
                 gamma, beta=1., w_mean=None, w_var=None,
                 w_real=None, T_real=None, lbd_real=None,
                 s1_mn=None, s1_var=None, nlbd=1e4):
        """
        dy, dc, dsc -- dimensionalities
        
        pi1 -- pi1 = P(c_1=1)
        tT, p00, p11 -- number of pseudo-observation and target
                        transition matrix for u_T
        tlbd, lbds -- number of pseudo-observation and target lambdas for u_lbd
        tnoise, noise_prec -- number of pseudo-observation and target precision of the noise
        
        s1_var-- the variance of a single style variable at time 1
        """
        self.dy, self.dc, self.dsc = dy, dc, dsc

        self.set_pi1(pi1)
        self.tT, self.p00, self.p11 = tT, p00, p11
        uT00, uT11 = tT*p00, tT*p11
        self.u_T = np.array([[uT00, tT-uT00],[tT-uT11, uT11]], dtype='d')
        self.update_u_T_tilde(self.u_T.copy())
        
        if s1_mn is None:
            s1_mn = np.zeros((dsc,), 'd')
        self.s1_mn = s1_mn
        self.set_s1var(s1_var)

        self.nlbd = nlbd

        self.tlbd, self.lbds0 = tlbd, lbds
        eta = np.zeros((dsc,), dtype='d')
        u_lbd = np.zeros((dsc,3), dtype='d')
        for j in range(dsc):
            eta[j] = tlbd/2.
            u_lbd[j,:] = -tlbd/2., tlbd*lbds[j], -tlbd/2.
        self.eta = np.repeat([eta], dc, axis=0)
        self.u_lbd = np.repeat([u_lbd], dc, axis=0)
        self.update_u_lbd_tilde(self.eta.copy(), self.u_lbd.copy())
        self.Z = self.Z_tilde.copy()

        # initialize input noise variables
        self.init_noise_prior(tnoise, noise_prec)
        # mean precision in the prior
        self.rho_mn0 = self.rho_mn.copy()

        if type(gamma) is float:
            self.gamma0 = gamma
            self.gamma = np.zeros((dc,dsc), dtype='d') + gamma
        else: # array
            self.gamma = gamma

        # annealing factor
        self.beta = beta

        # initialize weights variables
        if w_mean is None:
            self.w_mean = np.zeros((dy,dc,dsc), dtype='d')
        else:
            self.w_mean = w_mean
            
        if w_var is None:
            self.w_var = np.zeros((dy,dc,dsc), dtype='d') + 1./gamma
        else:
            self.w_var = w_var
            
        self.init_wDw()

        self.w_real = w_real
        self.T_real = T_real
        if lbd_real is not None and len(lbd_real.shape)==1:
            self.lbd_real = lrep(lbd_real, dc)
        else:
            self.lbd_real = lbd_real

    def init_noise_prior(self, tnoise, noise_prec):
        # initialize input noise variables
        self.tnoise, self.noise_prec = tnoise, noise_prec
        # u_noise[k,:] = [d, e]
        self.u_noise = np.zeros((self.dy,2), dtype='d')
        self.u_noise[:,0] = tnoise/2.
        self.u_noise[:,1] = (tnoise/2.)/noise_prec
        self.update_u_noise_tilde(self.u_noise)

    def init_wDw(self):
        dy, dc, dsc = self.dims()
        w_mean = self.w_mean
        w_var = self.w_var
        rho = self.rho_mn
        
        # <W_a^T D^-1 W_a>
        wDw = np.zeros((dc,dc,dsc,dsc), dtype='d')
        for a in range(dc):
            for i in range(dc):
                # <W_a>^T <D^-1> <W_i>
                #wDw[a,i,:,:] = mult(mult_diag(rho, w_mean[:,a,:].T, left=False),
                #                    w_mean[:,i,:])
                wDw[a,i,:,:] = gemm(1., mult_diag(rho, w_mean[:,a,:].T, left=False),
                                    w_mean[:,i,:])
                if a==i:
                    for j in range(dsc):
                        # <w_aj^T D^-1 w_aj> = trace(D^-1 Sigma_aj) + <w_aj^T> D^-1 <w_aj>
                        wDw[a,a,j,j] += (rho*w_var[:,a,j]).sum()

        self.wDw = wDw

    def set_pi1(self, pi1):
        self.pi1 = pi1
        self.pi = np.array([1.-pi1, pi1])

    def update_u_T_tilde(self, u_T_tilde):
        self.u_T_tilde = u_T_tilde
        u0_T_tilde = u_T_tilde.sum(axis=1)
        
        self.T_mn = np.zeros((2,2), dtype='d')
        self.logT_mn = np.zeros((2,2), dtype='d')
        for a in [0,1]:
            self.T_mn[a,:] = u_T_tilde[a,:]/u0_T_tilde[a]
            self.logT_mn[a,:] = special.digamma(u_T_tilde[a,:]) - \
                                special.digamma(u0_T_tilde[a])

    def _integrate(self, int_fun, lbd_max, h):
        #eps = 1e-3
        #pts = np.linspace(lbd_max-eps, lbd_max+eps, 100)
        #pts = np.where(pts<0., 0., pts)
        #pts = np.where(pts>1., 1., pts)
        pts = [lbd_max]
        res, err = sint.quad(int_fun, 0, 1.-h,
                             points=pts, epsabs=1e-12, epsrel=1e-12, limit=int(1e4)) #?1e5
        return res

    def update_u_lbd_tilde(self, eta_tilde, u_lbd_tilde):
        dy, dc, dsc = self.dims()

        self.eta_tilde = eta_tilde
        self.u_lbd_tilde = u_lbd_tilde

        h = 1./self.nlbd
        lbd = np.arange(0.,1.,h)
        lbd2 = lbd**2.
        sgm = 1.-lbd2

        Z_tilde = np.zeros((dc, dsc), dtype='d')
        sgms_mn = np.zeros((dc, dsc), dtype='d')
        sgms_inv_mn = np.zeros((dc, dsc), dtype='d')
        lbd_mn = np.zeros((dc, dsc), dtype='d')
        lbd_sgms_inv_mn = np.zeros((dc, dsc), dtype='d')
        lbd2_sgms_inv_mn = np.zeros((dc, dsc), dtype='d')
        log_sgm_mn = np.zeros((dc, dsc), dtype='d')
        for i in range(self.dc):
            for j in range(self.dsc):
                a,b,c = u_lbd_tilde[i,j,:]
                eta = eta_tilde[i,j]
                
                crv = -2*eta*lbd**3. + b*lbd2 + 2*(eta+a+c)*lbd + b
                lbd_max = lbd[np.argmin(crv>0)]
                #print 'lbd_max', lbd_max
               
                h=1e-3
                int_fun = _IntFun(lambda l,s: 1., eta, a,b,c, 1.)
                Qsum = self._integrate(int_fun, lbd_max, h)
                Z_tilde[i,j] = Qsum
                #print Qsum
                
                if Qsum<1e-300:
                    print '!!!! Emergency intervention in update_u_lbd_tilde',i,j,
                    print 'Qsum', Qsum
                    print '!!!! use point estimate of the mode'
                    sgm_max = 1-lbd_max**2.
                    sgms_mn[i,j] = sgm_max
                    sgms_inv_mn[i,j] = 1./sgm_max
                    lbd_mn[i,j] = lbd_max
                    lbd_sgms_inv_mn[i,j] =lbd_max/sgm_max
                    lbd2_sgms_inv_mn[i,j] =lbd_max**2./sgm_max
                    log_sgm_mn[i,j] = log(sgm_max)
                    continue
                    
                # <sgm>_QB
                int_fun = _IntFun(lambda l,s: s, eta, a,b,c, Qsum)
                sgms_mn[i,j] = self._integrate(int_fun, lbd_max, h)

                # <1./sgm>_QB
                int_fun = _IntFun(lambda l,s: 1./s, eta, a,b,c, Qsum)
                sgms_inv_mn[i,j] = self._integrate(int_fun, lbd_max, h)

                # <lbd>_QB
                int_fun = _IntFun(lambda l,s: l, eta, a,b,c, Qsum)
                lbd_mn[i,j] = self._integrate(int_fun, lbd_max, h)

                # <lbd/sgm>_QB
                int_fun = _IntFun(lambda l,s: l/s, eta, a,b,c, Qsum)
                lbd_sgms_inv_mn[i,j] = self._integrate(int_fun, lbd_max, h)

                # <lbd^2/sgm>_QB
                int_fun = _IntFun(lambda l,s: (l*l)/s, eta, a,b,c, Qsum)
                lbd2_sgms_inv_mn[i,j] = self._integrate(int_fun, lbd_max, h)

                # <log(sgm)/sgm>_QB
                int_fun = _IntFun(lambda l,s: log(s), eta, a,b,c, Qsum)
                log_sgm_mn[i,j] = self._integrate(int_fun, lbd_max, h)

        self.Z_tilde = Z_tilde
        self.sgms_mn = sgms_mn
        self.sgms_inv_mn = sgms_inv_mn
        self.lbd_mn = lbd_mn
        self.lbd_sgms_inv_mn = lbd_sgms_inv_mn
        self.lbd2_sgms_inv_mn = lbd2_sgms_inv_mn
        self.log_sgm_mn = log_sgm_mn

    def update_u_noise_tilde(self, u_noise_tilde):
        self.u_noise_tilde = u_noise_tilde
        self.rho_mn = u_noise_tilde[:,0]/u_noise_tilde[:,1]
        self.log_rho_mn = special.digamma(u_noise_tilde[:,0]) \
                          - log(u_noise_tilde[:,1])

    def set_s1var(self, var=None):
        if var==None: var=1.
        self.s1_var = var
        self.s1_var_mtx = np.eye(self.dsc, dtype='d')*var

    def dims(self):
        return self.dy, self.dc, self.dsc

    def sample_content(self, T, use_real=True):
        if use_real: ttr = self.T_real
        else: ttr = self.T_mn
        return hmm.sample_chain(T, self.dc, self.pi, ttr)

    def sample_style(self, T, use_real=True):
        dy, dc, dsc = self.dims()
        
        if use_real:
            lbd = self.lbd_real
            sgm_s_mtx = np.zeros((dc,dsc,dsc), dtype='d')
            for i in range(dc):
                sgm_s_mtx[i,:,:] = diag(1.-lbd[i,:]**2.)
        else:
            lbd = self.lbd_mn
            sgm_s_mtx = np.zeros((dc,dsc,dsc), dtype='d')
            for i in range(dc):
                sgm_s_mtx[i,:,:] = diag(self.sgms_mn[i,:])

        s = np.zeros((T, dc, dsc), dtype='d')
        for i in range(dc):
            s[:,i,:] = kalman.sample_LDS(T, diag(lbd[i,:]), 0.,
                                         self.s1_var_mtx, sgm_s_mtx[i,:,:])
        return s

    def sample_all(self, T, noise=False, s_off=True, use_real=True):
        # if noise==True, generate output with noise
        c = self.sample_content(T, use_real)
        s = self.sample_style(T, use_real)
        if use_real: w = self.w_real
        else: w = self.w_mean
        
        y = common.bimult(w, c, s)
        if noise:
            if np.isscalar(noise):
                sgm_y = np.array([noise]*self.dy)
            else: sgm_y = sqrt(1./self.rho_mn0) # use mean precision in the prior
            # generate noise, i-column has std sgm_y[i]
            y += normal(size=(T, self.dy))*sgm_y

        if s_off:
            # s is zero if content off
            for i in range(self.dc):
                s[:,i,:] *= np.repeat(c[:,i,np.newaxis], (self.dsc,), 1)

        return y, c, s
    

class State(object):
    def __init__(self, y, params):
        self.y = y
        self.params = params
        
        dy, dc, dsc = params.dims()
        T = y.shape[0]

        # inferred mean of the distribution of s for c=1
        self.mu = np.zeros((T, dc, dsc), dtype='d')
        # inferred covariance of the distribution of s for c=1
        self.sigma = np.zeros((T, dc, dsc, dsc), dtype='d')
        # <s_t^(a)>_Q
        self.s_mn = np.zeros((T, dc, dsc), dtype='d')
        # <s_t^(a) s_t^(a)^T>
        self.ss = np.zeros((T, dc, dsc, dsc), dtype='d')

        # inferred P(c_{a,t]=1)
        self.pc = np.zeros((T, dc), dtype='d')

    def set_to(self, c, s):
        dy, dc, dsc = self.params.dims()
        T = self.y.shape[0]
        
        self.pc = c.copy()
        self.mu = s.copy()
        sigma = np.zeros((T, dc, dsc ,dsc), dtype='d')
        for t in range(T):
            for i in range(dc):
                sigma[t,i,:,:] = np.eye(dsc, dtype='d')*1e-10
        self.sigma = sigma
        self.s_mn = self.mu.copy()
        self.s_cov = self.sigma.copy()
        self.sigma_mumu = np.zeros((T, dc, dsc, dsc), 'd')
        for t in range(T):
            for i in range(dc):
                mumu = outer(self.mu[t,i,:], self.mu[t,i,:])
                self.sigma_mumu[t,i,:,:] = self.sigma[t,i,:,:] + mumu
        self.ss = self.sigma_mumu.copy()
 
    def set_to_prior(self):
        dy, dc, dsc = self.params.dims()
        T = self.y.shape[0]
        
        # content
        pc, tmp1, tmp2 = hmm.get_prior(T, self.params.pi, self.params.T_mn)
        self.pc = np.repeat(pc[:,1:2], (dc,), 1)
        # style
        tmp = np.repeat(self.params.s1_mn[np.newaxis,:], (dc,), 0)
        self.mu = np.repeat(tmp[np.newaxis,:], (T,), 0)
        tmp = np.repeat(self.params.s1_var_mtx[np.newaxis,:], (dc,), 0)
        self.sigma = np.repeat(tmp[np.newaxis,:], (T,), 0)
        self.s_mn = self.mu.copy()
        self.s_cov = self.sigma.copy()
        self.sigma_mumu = np.zeros((T, dc, dsc, dsc), 'd')
        for t in range(T):
            for i in range(dc):
                mumu = outer(self.mu[t,i,:], self.mu[t,i,:])
                self.sigma_mumu[t,i,:,:] = self.sigma[t,i,:,:] + mumu
        self.ss = self.sigma_mumu.copy()

    def get_epsilon(self):
        T = self.y.shape[0]
        pc = self.pc
        
        epsilon = np.zeros((T-1,self.params.dc,2,2), dtype='d')
        epsilon[:,:,1,1] = pc[:-1,:]*pc[1:,:]
        epsilon[:,:,1,0] = pc[:-1,:]*(1.-pc[1:,:])
        epsilon[:,:,0,1] = (1.-pc[:-1,:])*pc[1:,:]
        epsilon[:,:,0,0] = (1.-pc[:-1,:])*(1.-pc[1:,:])
    
        return epsilon
