import numpy as np
from numpy import dot, exp, log
from numpy.linalg import inv

from structured.common import bimult, gemm


def sample_c(y, c, s, params, lhood = None):
    dy, dc, dsc = params.dims()
    T = c.shape[0]

    logT = params.logT_mn
    rho_mn = params.rho_mn
    w_mean = params.w_mean

    c = c.copy()

    permc = np.random.permutation(dc)
    random_numbers  = np.random.rand(T, dc)
    m = y - bimult(w_mean, c, s)
    
    for t in range(T):
        for a in permc:
            logp = np.zeros((2,))
            m_ta = [0., 0.]

            # case 1: c[t,a] does not change
            c_ta = int(c[t,a])
            m_ta[c_ta] = m[t,:]
            
            logp[c_ta] = -0.5* dot(m_ta[c_ta].T, rho_mn*m_ta[c_ta])
            if t==0: logp[c_ta] += log(params.pi[c_ta])
            else: logp[c_ta] += logT[c[t-1,a],c_ta]
            if t<T-1: logp[c_ta] += logT[c_ta,c[t+1,a]]

            # case 2: c[t,a] does change
            c_ta = int(1. - c[t,a])
            if c_ta==1:
                m_ta[c_ta] = m[t,:] - dot(w_mean[:,a,:], s[t,a,:])
            else:
                m_ta[c_ta] = m[t,:] + dot(w_mean[:,a,:], s[t,a,:])
            
            logp[c_ta] = -0.5* dot(m_ta[c_ta].T, rho_mn*m_ta[c_ta])
            if t==0: logp[c_ta] += log(params.pi[c_ta])
            else: logp[c_ta] += logT[c[t-1,a],c_ta]
            if t<T-1: logp[c_ta] += logT[c_ta,c[t+1,a]]

            # P(c_ta = 1 | rest)
            logp -= logp.max()
            prob = exp(logp) / exp(logp).sum()
            # sample c_ta
            c[t,a] = float(prob[1] > random_numbers[t,a])
            m[t,:] = m_ta[int(c[t,a])]
    
    return c


def sample_s(y, c, s, params):
    dy, dc, dsc = params.dims()
    T = c.shape[0]

    rho_mn = params.rho_mn
    w_mean = params.w_mean
    wDw = params.wDw
    drg = np.arange(dsc)
    s1_var = params.s1_var
    lbd2_sgms_inv_mn, lbd_sgms_inv_mn = params.lbd2_sgms_inv_mn, params.lbd_sgms_inv_mn
    sgms_inv_mn = params.sgms_inv_mn

    s = s.copy()

    permc = np.random.permutation(dc)

    for t in range(T):
        for a in permc:
            if t==0:
                # Q^B(s_0^a)
                # sgm_B_inv[a,b] = (sgm_a^B)_bb
                sgmB_inv = 1./s1_var + lbd2_sgms_inv_mn[a,:]
                muB_sgmB_inv = s[t+1:t+2,a,:] * lbd_sgms_inv_mn[a,:]
            elif t==T-1:
                # Q^B(s_T^a)
                sgmB_inv = sgms_inv_mn[a,:]
                muB_sgmB_inv =  s[t-1:t,a,:] * lbd_sgms_inv_mn[a,:]
            else:
                # Q^B(s_s^a)
                sgmB_inv = sgms_inv_mn[a,:] + lbd2_sgms_inv_mn[a,:]
                tmp = s[t-1:t,a,:]+s[t+1:t+2,a,:]
                muB_sgmB_inv = tmp * lbd_sgms_inv_mn[a,:]
                
            ## new distribution
            # covariance
            # inv_sigma = diag(sgmB_inv) + wDw[a,a,:,:]
            inv_sigma = wDw[a,a,:,:].copy()
            inv_sigma[drg,drg] += sgmB_inv
            sigma = inv(inv_sigma)

            # mean
            tmp = 0.
            for i in range(dc):
                if i!=a: tmp += dot(w_mean[:,i,:], c[t,i]*s[t,i,:])
            # mu_sigma_inv = muB_sgmB_inv + (yt-tmp)^T <D^-1> <W_a>
            mu_sigma_inv = muB_sgmB_inv + \
                           dot(rho_mn*(y[t:t+1,:]-tmp), w_mean[:,a,:])
            #mu_a = dot(mu_sigma_inv, sigma_a).T
            mean = gemm(1., mu_sigma_inv, sigma)[0,:]
            
            # sample from gaussian
            s[t,a,:] = np.random.multivariate_normal(mean, sigma)

    return s

