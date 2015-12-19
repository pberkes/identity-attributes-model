import numpy as np
from numpy import dot, exp, log, outer, trace
from numpy.linalg import det, inv
import scipy.special as special

from structured import common
from structured.common import gemm, log0, mult_diag


def compute_Q_A(t, a, state, params, causal=False):
    # compute Q_ra^A(c_ra)
    
    pc = state.pc
    logT = params.logT_mn

    if causal:
        # ?? subtract max to make this more robust
        if t==0:
            Q_A = params.pi
        else:
            # Q^A(a,r)
            Q_A = exp((1.-pc[t-1,a])*logT[0,:] + pc[t-1,a]*logT[1,:])
        Q_A /= Q_A.sum()
        return Q_A

    if t==0:
        # Q^A(a,0)
        log_Q_A = log(params.pi) + \
                  (1.-pc[1,a])*logT[:,0] + pc[1,a]*logT[:,1]
    elif t==state.y.shape[0]-1:
        # Q^A(a,T)
        log_Q_A = (1.-pc[t-1,a])*logT[0,:] + pc[t-1,a]*logT[1,:]
    else:
        # Q^A(a,r)
        log_Q_A = (1.-pc[t-1,a])*logT[0,:] + pc[t-1,a]*logT[1,:] +\
                  (1.-pc[t+1,a])*logT[:,0] + pc[t+1,a]*logT[:,1]

    # normalize
    Q_A = exp(log_Q_A - log_Q_A.max())
    Q_A /= Q_A.sum()
    return Q_A


def compute_Q_B(t, a, state, params, causal=False):
    # compute Q_ra^B(s_ra)
    
    s_mn = state.s_mn

    if causal:
        if t==0:
            sgmB_inv = np.ones((params.dsc,), dtype='d')
            muB_sgmB_inv = np.zeros((1,params.dsc), dtype='d')
            muB = np.zeros((params.dsc,1), dtype='d')
        else:
            # Q^B(s_s^a)
            sgmB_inv = params.sgms_inv_mn[a,:]
            muB_sgmB_inv = s_mn[t-1:t,a,:] * params.lbd_sgms_inv_mn[a,:]
            muB = (muB_sgmB_inv / sgmB_inv).T
        return muB, muB_sgmB_inv, sgmB_inv

    if t==0:
        # Q^B(s_0^a)
        # sgm_B_inv[a,b] = (sgm_a^B)_bb
        sgmB_inv = 1./params.s1_var + params.lbd2_sgms_inv_mn[a,:]
        muB_sgmB_inv = s_mn[t+1:t+2,a,:] * params.lbd_sgms_inv_mn[a,:]
        muB = (muB_sgmB_inv / sgmB_inv).T
    elif t==state.y.shape[0]-1:
        # Q^B(s_T^a)
        sgmB_inv = params.sgms_inv_mn[a,:]
        muB_sgmB_inv =  s_mn[t-1:t,a,:] * params.lbd_sgms_inv_mn[a,:]
        muB = (muB_sgmB_inv / sgmB_inv).T
    else:
        # Q^B(s_s^a)
        sgmB_inv = params.sgms_inv_mn[a,:] + params.lbd2_sgms_inv_mn[a,:]
        tmp = s_mn[t-1:t,a,:]+s_mn[t+1:t+2,a,:]
        muB_sgmB_inv = tmp * params.lbd_sgms_inv_mn[a,:]
        muB = (muB_sgmB_inv / sgmB_inv).T

    return muB, muB_sgmB_inv, sgmB_inv


def e_step(state, params, permute=False, only_this=None, causal=False, only_last=False):
    # useful quantities
    y = state.y
    T = y.shape[0]
    dy, dc, dsc = params.dims()
    
    rho_mn = params.rho_mn
    w_mean, w_var = params.w_mean, params.w_var
    
    sigma, mu = state.sigma, state.mu
    pc = state.pc
    s_mn = state.s_mn
    ss, s_cov = state.ss, state.s_cov

    sgmB = np.zeros((T,dc,dsc), dtype='d')
    state.sgmB = sgmB
    # sigma_mumu = sigma + mu * mu^T
    sigma_mumu = np.zeros((T, dc, dsc, dsc), 'd')
    state.sigma_mumu = sigma_mumu

    wDw = params.wDw

    drg = np.arange(dsc)

    if only_this is not None:
        permc = np.array([only_this], dtype='i')

    if only_last:
        range_T = [T-1]
    else:
        range_T = range(T)

    for t in range_T:
        # cycle over all contents in random order
        if permute:
            permc = np.random.permutation(dc)
        else:
            permc = np.arange(dc)
        for a in permc:
            # Q_ra^A(c_ra)
            Q_A = compute_Q_A(t, a, state, params, causal)

            # Q_ra^B(s_ra)
            muB, muB_sgmB_inv, sgmB_inv = compute_Q_B(t, a, state, params, causal)
            
            ## new distribution
            # covariance
            # inv_sigma = diag(sgmB_inv) + wDw[a,a,:,:]
            inv_sigma = params.beta*wDw[a,a,:,:].copy()
            inv_sigma[drg,drg] += sgmB_inv
            sigma_a = inv(inv_sigma)
            sigma[t,a,:,:] = sigma_a

            # mean
            tmp = 0.
            for i in range(dc):
                if i!=a: tmp += dot(w_mean[:,i,:], pc[t,i]*mu[t,i,:])
            # mu_sigma_inv = muB_sgmB_inv + (yt-tmp)^T <D^-1> <W_a>
            mu_sigma_inv = muB_sgmB_inv + \
                           params.beta*dot(rho_mn*(y[t:t+1,:]-tmp), w_mean[:,a,:])
            #mu_a = dot(mu_sigma_inv, sigma_a).T
            mu_a = gemm(1., mu_sigma_inv, sigma_a).T
            mu[t,a,:] = mu_a[:,0]

            # constants
            log_det_B = -log(sgmB_inv).sum()
            if det(sigma_a)<0.:
                print '@@@@@@@@@@@@@@@@@@', det(sigma_a),t,a
            B = exp(-0.5*(log_det_B - log(det(sigma_a))
                          + dot(muB_sgmB_inv, muB)
                          - dot(mu_sigma_inv, mu_a) ))

            # ZA is ZA = 1./(p_tld[0] + B*p_tld[1])
            # the following is more stable (handles B=inf)
            ZAB = 1./(1./B * Q_A[0] + Q_A[1])

            # Q(c_{a,s}=1)
            pc[t,a] = ZAB*Q_A[1]

            ## store statistics

            # s_mn
            s_mn[t,a,:] = pc[t,a]*mu_a[:,0] + (1.-pc[t,a])*muB[:,0]
            
            # ss
            #ss[t,a,:,:] = pc[t,a]*(sigma_a+outer(mu[t,a,:], mu[t,a,:])) +\
            #              (1.-pc[t,a])*(sgmB+outer(muB, muB))
            sgmB[t,a,:] = 1./sgmB_inv
            #sgmB_muBmuBT = dot(muB, muB.T)
            sgmB_muBmuBT = gemm(1., muB, muB, trans_b=1)
            sgmB_muBmuBT[drg,drg] += sgmB[t,a,:]
            #mua_muaT = dot(mu[t:t+1,a,:].T, mu[t:t+1,a,:])
            mua_muaT = gemm(1., mu[t:t+1,a,:], mu[t:t+1,a,:], trans_a=1)
            sigma_mumu[t,a,:,:] = sigma_a+mua_muaT
            ss[t,a,:,:] = pc[t,a]*(sigma_mumu[t,a,:,:]) +\
                          (1.-pc[t,a])*(sgmB_muBmuBT)

            # s_cov
            # smn_smnT = dot(s_mn[t:t+1,a,:].T, s_mn[t:t+1,a,:])
            smn_smnT = gemm(1., s_mn[t:t+1,a,:], s_mn[t:t+1,a,:], trans_a=1)
            s_cov[t,a,:,:] = ss[t,a,:,:] - smn_smnT

def close_e_step(params, state):
    # ## compute some useful quantities at the end of the e-step
    dy, dc, dsc = params.dims()
    pc, mu, sigma_mumu = state.pc, state.mu, state.sigma_mumu
    y = state.y
    T = y.shape[0]
        
    # compute ycs[a,:,:] = sum_t y_t <s_ta c_ta>^T
    # ycs[k,a,b] is r_{kab} in the paper
    ycs = np.zeros((dy, dc, dsc), 'd')
    
    # compute cscs[i,a,:,:] = sum_t <s_ti c_ti> <s_ta c_ta>^T
    # cscs[i,a,j,b] is R_{ijab} in the paper
    cscs = np.zeros((dc, dc, dsc, dsc), 'd')
    
    for a in range(dc):
        for t in range(T):
            #ycs[:,a,:] += dot(y[t:t+1,:].T, pc[t,a]*mu[t:t+1,a,:])
            ycs[:,a,:] += gemm(1., y[t:t+1,:], pc[t,a]*mu[t:t+1,a,:], trans_a=1)
            for i in range(dc):
                if i==a:
                    cscs[a,a,:,:] += pc[t,a]*sigma_mumu[t,a,:,:]
                else:
                    # !!!! different from model.varapprox
                    #cscs[i,a,:,:] += dot(pc[t,i]*mu[t:t+1,i,:].T,
                    #                      pc[t,a]*mu[t:t+1,a,:])
                    cscs[i,a,:,:] += gemm(1., pc[t,i]*mu[t:t+1,i,:],
                                          pc[t,a]*mu[t:t+1,a,:], trans_a=1)

    state.ycs = ycs
    state.cscs = cscs

def m_step_T(state, params):
    T = state.y.shape[0]
    u_T = params.u_T
    pc = state.pc
    
    u_T_tilde = np.zeros((2,2), dtype='d')
    for a in range(2):
        if a==1: Q_a = pc[1:,:]
        else: Q_a = 1.-pc[1:,:]
        for b in range(2):
            if b==1: Q_b = pc[:T-1,:]
            else: Q_b = 1.-pc[:T-1,:]
            u_T_tilde[a,b] = u_T[a,b] + (Q_a*Q_b).sum()
    params.update_u_T_tilde(u_T_tilde)

def m_step_lbd(state, params):
    dy, dc, dsc = params.dims()
    T = state.y.shape[0]
    eta = params.eta
    u_lbd = params.u_lbd
    ss, s_mn = state.ss, state.s_mn
    
    eta_tilde = np.zeros((dc,dsc), dtype='d')
    u_lbd_tilde = np.zeros((dc, dsc, 3), dtype='d')
    if T>500: # hard limit to work around numerical issues
        T = 500
        ss = ss[:500,:,:,:].copy()
        s_mn = s_mn[:500,:,:].copy()
    for a in range(dc):
        for b in range(dsc):
            eta_tilde[a,b] = eta[a,b] + (T-1.)/2.
            u_lbd_tilde[a,b,0] = u_lbd[a,b,0] - 0.5*ss[1:,a,b,b].sum()
            u_lbd_tilde[a,b,1] = u_lbd[a,b,1] + \
                                 (s_mn[:T-1,a,b]*s_mn[1:,a,b]).sum()
            u_lbd_tilde[a,b,2] = u_lbd[a,b,2] - 0.5*ss[:T-1,a,b,b].sum()

    params.update_u_lbd_tilde(eta_tilde, u_lbd_tilde)

def m_step_rho(state, params):
    dy, dc, dsc = params.dims()
    y = state.y
    w_mean = params.w_mean
    cscs = state.cscs
    u_noise_tilde = params.u_noise.copy()
    
    # d' = d+T/2
    u_noise_tilde[:,0] += params.beta*y.shape[0]/2.

    # e'
    tmp = 0.
    for i in range(dc):
        for j in range(dsc):
            for m in range(dc):
                for n in range(dsc):
                    tmp += w_mean[:,i,j]*w_mean[:,m,n]*cscs[i,m,j,n]
                    
    u_noise_tilde[:,1] += params.beta*0.5*((y*y).sum(axis=0)
                                           -2.*(w_mean*state.ycs).sum(axis=1).sum(axis=1)
                                           +tmp)

    params.update_u_noise_tilde(u_noise_tilde)
             

def m_step(state, params, tol=1e-10, max_iter=200, learn_rho=True, only_this=None):
    # M-Step for T
    m_step_T(state, params)
    # M-Step for lambda
    m_step_lbd(state, params)

    # M-Step for rho
    if learn_rho:
        m_step_rho(state, params)

    # M-Step for W
    
    # useful quantities
    dy, dc, dsc = params.dims()
    w_mean, w_var = params.w_mean, params.w_var
    rho, gamma = params.rho_mn, params.gamma

    ycs, cscs = state.ycs, state.cscs

    # iterate until convergence or max_iter
    N = 0
    while True:
        N += 1
        max_change = 0.

        # save old values
        if only_this is None:
            contents = np.arange(dc)
        else:
            contents = np.array([only_this], dtype='i')
            
        for a in contents:
            for b in range(dsc):
                # save old values
                w_mean_old = w_mean[:,a,b].copy()
                w_var_old = w_var[:,a,b].copy()

                # compute Sigma_ab^W (w_var)
                w_var[:,a,b] = 1./(params.beta*rho*cscs[a,a,b,b] + gamma[a,b])

                # compute mu_ab^W (w_mean)
                # second term in the mean expression
                num = 0.
                for i in range(dc):
                    for j in range(dsc):
                        if i!=a or j!=b:
                            num += cscs[i,a,j,b]*w_mean[:,i,j]

                w_mean[:,a,b] = w_var[:,a,b]*params.beta*rho*(ycs[:,a,b] - num)

                # compute KL divergence
                dmean = w_mean_old-w_mean[:,a,b]
                kl = 0.5*(log(w_var_old).sum()
                          -log(w_var[:,a,b]).sum()
                          +(w_var[:,a,b]/w_var_old).sum()
                          +(dmean*dmean/w_var_old).sum()
                          -dy)
                
                max_change = max(max_change, kl)

        #print 'max_change', max_change
        if max_change<tol:
            print 'M-Step took %d iterations' % N
            break
        if N>max_iter:
            # ?? change into a warning
            str = '\nM-Step failed to converge, max_change=%f\n' % max_change
            print str
            break
            #raise Exception, str

    # reload rho after modification
    rho = params.rho_mn
    
    # <W_a^T D^-1 W_a>
    wDw = np.zeros((dc,dc,dsc,dsc), dtype='d')
    for a in range(dc):
        for i in range(dc):
            # <W_a>^T <D^-1> <W_i>
            #wDw[a,i,:,:] = dot(mult_diag(rho, w_mean[:,a,:].T, left=False),
            #                    w_mean[:,i,:])
            wDw[a,i,:,:] = gemm(1., mult_diag(rho, w_mean[:,a,:].T, left=False),
                                w_mean[:,i,:])
            if a==i:
                for j in range(dsc):
                    # <w_aj^T D^-1 w_aj> = trace(D^-1 Sigma_aj) + <w_aj^T> D^-1 <w_aj>
                    wDw[a,a,j,j] += (rho*w_var[:,a,j]).sum()
    params.wDw = wDw

def learn_gamma(params):
    dy, dc, dsc = params.dims()
    for a in range(dc):
        for b in range(dsc):
            params.gamma[a,b]=dy/((params.w_var[:,a,b]).sum() +
                                  (params.w_mean[:,a,b]**2.).sum())

def free_energy(params, state):
    # useful quantities
    dy, dc, dsc = params.dims()

    PI = np.pi
    log_2pi = log(2.*PI)
    T = state.y.shape[0]

    mu = state.mu
    smn = state.s_mn
    s1_var = params.s1_var

    w_mn = params.w_mean
    w_var = params.w_var
    sgm = state.sigma
    sgmB = state.sgmB
    gamma = params.gamma

    uTtld = params.u_T_tilde
    ulbdtld = params.u_lbd_tilde
    etatld = params.eta_tilde
    Ztld = params.Z_tilde

    uT = params.u_T
    ulbd = params.u_lbd
    eta = params.eta
    Z = params.Z

    log_sgm_mn = params.log_sgm_mn
    sgms_inv_mn = params.sgms_inv_mn
    lbd_sgms_inv_mn = params.lbd_sgms_inv_mn
    lbd2_sgms_inv_mn = params.lbd2_sgms_inv_mn

    # pc = [1.-c, c]
    QC = np.zeros((T, dc, 2), 'd')
    QC[:,:,0] = 1.-state.pc
    QC[:,:,1] = state.pc.copy()

    cscs = state.cscs
    ss = state.ss
    wDw = params.wDw
    rho_mn = params.rho_mn
    log_rho_mn = params.log_rho_mn

    ###
    ### H(Q(C, S))
    HQCS = 0.
    for t in range(T):
        for a in range(dc):
            HQCS -= QC[t,a,1]*(log0(QC[t,a,1])
                               - dsc/2. - dsc/2.*log_2pi - 0.5*log0(det(sgm[t,a,:,:])))
            HQCS -= QC[t,a,0]*(log0(QC[t,a,0])
                               - dsc/2. - dsc/2.*log_2pi - 0.5*log0(sgmB[t,a,:]).sum())

    ### H(Q(T))
    HQT = 0.
    for i in range(2):
        HQT += +special.betaln(uTtld[i,0], uTtld[i,1]) \
               -(uTtld[i,0]-1.)*params.logT_mn[i,0] \
               -(uTtld[i,1]-1.)*params.logT_mn[i,1]

    ### H(Q(Lambda))
    HQL = 0.
    for a in range(dc):
        for b in range(dsc):
            at, bt, ct = ulbdtld[a,b,:]
            HQL += log0(Ztld[a,b]) + etatld[a,b]*log_sgm_mn[a,b] \
                   - at*sgms_inv_mn[a,b] - bt*lbd_sgms_inv_mn[a,b] \
                   - ct*lbd2_sgms_inv_mn[a,b]

    ### <log P(C|T)>
    log_PC = 0.
    # t=0
    for a in range(dc):
        log_PC = (QC[0,a,:]*log(params.pi)).sum()
    # t>0
    for t in range(1,T):
        for a in range(dc):
            log_PC += (outer(QC[t-1,a,:],QC[t,a,:])*params.logT_mn).sum()
    
    ### <log P(S|Lambda)>
    log_PS = 0.
    for a in range(dc):
        # t=0
        log_PS += -dsc/2.*log(2.*PI*s1_var) \
                  -0.5/s1_var*trace(ss[0,a,:,:])
        # t>0
        log_PS += -(T-1.)*dsc/2.*log_2pi -(T-1.)/2.*log_sgm_mn[a,:].sum()
        for b in range(dsc):
            log_PS += -0.5*(
                sgms_inv_mn[a,b]*ss[1:,a,b,b].sum()
                -2.*lbd_sgms_inv_mn[a,b]*(smn[1:,a,b]*smn[:-1,a,b]).sum()
                +lbd2_sgms_inv_mn[a,b]*ss[:-1,a,b,b].sum()
                )

    ### <log P(T)>_Q(T)
    log_PT = 0.
    for i in range(2):
        log_PT += -special.betaln(uT[i,0], uT[i,1]) \
                  +(uT[i,0]-1.)*params.logT_mn[i,0] \
                  +(uT[i,1]-1.)*params.logT_mn[i,1]

    ### <log P(Lambda)>_Q(Lambda)
    log_PL = 0.
    for a in range(dc):
        for b in range(dsc):
            at, bt, ct = ulbd[a,b,:]
            log_PL += -log0(Z[a,b]) - eta[a,b]*log_sgm_mn[a,b] \
                      + at*sgms_inv_mn[a,b] + bt*lbd_sgms_inv_mn[a,b] \
                      + ct*lbd2_sgms_inv_mn[a,b]

    ### KL(W)
    KL_W_rho = 0.
    for a in range(dc):
        for b in range(dsc):
            KL_W_rho += 0.5*(-log0(gamma[a,b])-log0(w_var[:,a,b])
                             +gamma[a,b]*w_var[:,a,b]
                             -1.
                             +gamma[a,b]*w_mn[:,a,b]*w_mn[:,a,b]).sum()

    ### KL(rho)
    dt, et = params.u_noise_tilde[:,0], params.u_noise_tilde[:,1]
    d, e = params.u_noise[:,0], params.u_noise[:,1]
    KL_rho = (dt*log0(et) - d*log0(e)
              -special.gammaln(dt) + special.gammaln(d)
              +(dt-d)*(special.digamma(dt)-log0(et))
              -dt*(1.-e/et)).sum()
    
    ### <log P(Y|C,S,W)>_Q(C,S)Q(W)
    tmp = 0.
    for i in range(dc):
        for k in range(dc):
            #tmp += trace(dot(wDw[k,i,:,:], cscs[i,k,:,:]))
            tmp += trace(gemm(1., wDw[k,i,:,:], cscs[i,k,:,:]))

    # dot(state.y, diag(params.rho_mn))
    rho_y = mult_diag(rho_mn, state.y, left=False)

    log_PY = params.beta*\
             (-T*dy/2.*log_2pi +T/2.*log_rho_mn.sum() \
              -0.5 * ((rho_y * state.y).sum()
                      -2.*(rho_y*common.bimult(w_mn, QC[:,:,1], state.mu)).sum()
                      + tmp))

    #print 'HQCS,HQT,HQL',HQCS,HQT,HQL
    #print 'log_PC, log_PS', log_PC, log_PS
    #print 'log_PT, log_PL', log_PT, log_PL
    #print 'KL_W_rho, KL_rho', KL_W_rho, KL_rho
    #print 'log_PY', log_PY

    return HQCS + HQT + HQL \
           + log_PC + log_PS + log_PT + log_PL \
           - KL_W_rho - KL_rho + log_PY

# ##################################################
    
def ml_sigmay(params, state):
    # ?? needs to be adapted
    dy, dc, dsc = params.dims()
    T = state.y.shape[0]

    w_mn = params.w_mean
    w_var = params.w_var
    sigma, mu, pc = state.sigma, state.mu, state.pc
    y = state.y
    
    # sigma_mumu = sigma + mu * mu^T
    sigma_mumu = np.zeros((T, dc, dsc, dsc), 'd')
    for t in range(T):
        for a in range(dc):
            #mua_muaT = dot(mu[t:t+1,a,:].T, mu[t:t+1,a,:])
            mua_muaT = gemm(1., mu[t:t+1,a,:], mu[t:t+1,a,:], trans_a=1)
            sigma_mumu[t,a,:,:] = sigma[t,a,:,:] + mua_muaT
    
    # cscs[i,a,:,:] = sum_t <s_ti c_ti> <s_ta c_ta>^T
    cscs = np.zeros((dc, dc, dsc, dsc), 'd')
    for t in range(T):
        for a in range(dc):
            for i in range(dc):
                if i==a:
                    cscs[a,a,:,:] += pc[t,a]*sigma_mumu[t,a,:,:]
                else:
                    # !!!! different from model.varapprox
                    #cscs[i,a,:,:] += dot(pc[t,i]*mu[t:t+1,i,:].T,
                    #                      pc[t,a]*mu[t:t+1,a,:])
                    cscs[i,a,:,:] += gemm(1., pc[t,i]*mu[t:t+1,i,:],
                                          pc[t,a]*mu[t:t+1,a,:], trans_a=1)

    ml_sgm = np.zeros((dy,), dtype='d')
    for k in range(dy):
        sgmk = 0.

        sgmk += (y[:,k]*y[:,k]).sum()
        
        tmp = 0.
        for i in range(dc):
            for l in range(dsc):
                tmp += (w_mn[k,i,l]*pc[:,i]*mu[:,i,l])
        sgmk += -2.*(y[:,k]*tmp).sum()

        tmp = 0.
        for a in range(dc):
            for i in range(dc):
                #for j in range(dsc):
                #    for m in range(dsc):
                #        if a!=i or j!=m:
                #            tmp += w_mn[k,a,j]*w_mn[k,i,m]*cscs[i,a,m,j]
                #        else:
                #            tmp += (w_mn[k,a,j]*w_mn[k,a,j] + w_var[k,a,j])*cscs[i,a,m,j]
                tmp += (outer(w_mn[k,a,:],w_mn[k,i,:]).T*cscs[i,a,:,:]).sum()
                if a==i:
                    for j in range(dsc):
                        tmp += w_var[k,a,j]*cscs[a,a,j,j]
        sgmk += tmp

        ml_sgm[k] = 1./T*sgmk

    return ml_sgm
