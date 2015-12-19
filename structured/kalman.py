from mdp import utils
import numpy as np
from numpy.random import normal
from scipy.linalg import blas, inv


PI = np.pi
PI2 = 2*np.pi

# Faster routines
a = np.zeros((1,1), 'd')
gemm, = blas.get_blas_funcs(('gemm',),(a,a))


def sample_LDS(T, A, pi1, V1, Q):
    """
    Sample a random sequence from the prior.

    Arguments:
    A - x(t+1) = Ax(t) + w(t)
    pi1 - mean of v(1)
    V1 - covariance of v(1)
    Q - covariance of w(t)

    Return:
    x - sequence
    """

    dx = A.shape[0]
    # noise
    w = normal(0., 1., size=(T, dx))
    w = np.dot(w, np.real(utils.sqrtm(Q)).T)
    # latent variables
    x = np.zeros((T, dx), dtype='d')
    x[0,:] = normal(0., 1., size=(dx,))
    x[0,:] = np.dot(x[0,:], np.real(utils.sqrtm(V1)).T) + pi1
    for t in range(1, T):
        x[t,:] = np.dot(x[t-1,:], A.T) + w[t]
    return x


def execute_LDS(x, C, R=None):
    """
    Generate observations in a LDS.

    Arguments:
    x - states
    C - observation matrix
    R - covariance matrix of the observation noise
        if None, generation does not include noise

    Return:
    y - y(t) = Cx(t) + v(t)
    """
    y = np.dot(x, C.T)
    if R != None:
        v = normal(0., 1., size=y.shape)
        v = np.dot(v, np.real(utils.sqrtm(R)).T)
        y += v
    return y


def smooth(y, A, C, pi1, V1, Q, R):
    """
    Kalman smoother. It infers P(x(t)) from observations y(1)...y(T)
    for a LDS. Missing data can be marked using nan's.

    Arguments:
    A - x(t+1) = Ax(t) + w(t)
    C - y(t) = Cx(t) + v(t)
    pi1 - mean of v(1)
    V1 - covariance of v(1)
    Q - covariance of w(t) (dynamic noise)
    R(t) - covariance of v(t) (observation noise)
           If noise stationary noise, single matrix, else T matrices

    Return:
    x, V - mean and covariance of the states
    """

    T, dy = y.shape
    dx = A.shape[0]
    stat_noise = len(R.shape)==2

    # ### useful stuff

    # constant term in front of the log-likelihood
    const = -dy/2. * np.log(PI2)

    # ### forward recursion (filter)
    
    # ### variables definition
    # x_t^{t-1} = E(xt | y1 ... yt-1)
    xt_tm1 = np.zeros((T, dx), 'd')
    # x_t^{t} =  E(xt | y1 ... yt)
    xt_t = np.zeros((T, dx), 'd')
    # V_t^{t-1} = Var(xt | y1 ... yt-1)
    Vt_tm1 = np.zeros((T, dx, dx), 'd')
    # V_t^{t} = Var(xt | y1 ... yt)
    Vt_t = np.zeros((T, dx, dx), 'd')
    # Kalman gain
    Kt = np.zeros((T, dx, dy), 'd')

    xpre = pi1
    Vpre = V1
    for t in range(T):
        xt_tm1[t,:] = xpre
        Vt_tm1[t,:,:] = Vpre
        if stat_noise: Rt = R
        else: Rt = R[t,:,:]
        
        # ### compute the Kalman gain K
        # TODO Zoubin computes Kt in two different ways depending on
        #      the dimensionalities.
        # VC = np.dot(Vpre, tr(C))
        VC = gemm(1., Vpre, C, 0., None, 0, 1)
        # Sigma = np.dot(C, VC) + Rt
        Sigma = gemm(1., C, VC, 0., None, 0, 0) + Rt
        inv_Sigma = inv(Sigma)
        # Kalman gain
        K = gemm(1., VC, inv_Sigma, 0., None, 0, 0)
        
        # xpost := E(xt | y1 ... yt)
        if np.isnan(y[t,0]):
            # missing observation
            xpost = xpre.copy()
            Vpost = Vpre.copy()
        else:
            y_diff = y[t:t+1,:] - np.dot(xpre, C.T)
            # xpost = xpre + np.dot(y_diff, tr(K))[0,:]
            xpost = xpre + gemm(1., y_diff, K, 0., None, 0, 1)[0,:]
            # Vpost := Var(xt | y1 ... yt)
            # this is the equation in Gaharamani and Hinton:
            # Vpost = Vpre - np.dot(K, tr(VC))
            # the following way is much more stable (see M.Welling eq. 43)
            # I_KC = np.eye(dx) - np.dot(K, C)
            # Vpost = np.dot(I_KC, np.dot(Vpre, tr(I_KC))) + \
            #         np.dot(K, np.dot(Rt, tr(K)))
            I_KC = np.eye(dx) - gemm(1., K, C, 0., None, 0, 0)
            Vpost = gemm(1., I_KC, gemm(1., Vpre, I_KC, 0., None, 0, 1),
                         0., None, 0, 0) + \
                    gemm(1., K, gemm(1., Rt, K, 0., None, 0, 1),
                         0., None, 0, 0)

        # TODO do not compute this on the last iteration
        # xpre := E(xt+1 | y1 ... yt)
        xpre = np.dot(xpost, A.T)
        # Vpre := Var(xt+1 | y1 ... yt)
        # Vpre = np.dot(A, np.dot(Vpost, tr(A))) + Q
        Vpre = gemm(1., A, gemm(1., Vpost, A, 0., None, 0, 1),
                    0., None, 0, 0) + Q
        
        xt_t[t,:] = xpost
        Vt_t[t,:,:] = Vpost
        Kt[t,:,:] = K

    # ### backward recursion (smooth)

    # ### variables definition
    J = np.zeros((T-1, dx, dx), 'd')
    # x_t^T = E(xt | y1 ... yT)
    xt_T = np.zeros((T, dx), 'd')
    # V_t^T = Var(xt | y1 ... yT)
    Vt_T = np.zeros((T, dx, dx), 'd')

    xt_T[T-1,:] = xt_t[T-1,:]
    Vt_T[T-1,:,:] = Vt_t[T-1,:,:]
    for t in range(T-1, 0, -1):
        #J[t-1,:,:] = np.dot(Vt_t[t-1,:,:], np.dot(tr(A), inv(Vt_tm1[t,:,:])))
        J[t-1,:,:] = gemm(1., Vt_t[t-1,:,:],
                              gemm(1., A, inv(Vt_tm1[t,:,:]), 0., None, 1, 0),
                          0., None, 0, 0)
        xt_T[t-1,:] = xt_t[t-1,:] + \
                      np.dot(xt_T[t,:] - np.dot(xt_t[t-1,:], A.T),
                               J[t-1,:,:].T)
        #Vt_T[t-1,:,:] = Vt_t[t-1,:,:] + \
        #                np.dot(J[t-1,:,:], np.dot(Vt_T[t,:,:] - Vt_tm1[t,:,:],
        #                                      tr(J[t-1,:,:])))
        Vt_T[t-1,:,:] = Vt_t[t-1,:,:] + \
                        gemm(1., J[t-1,:,:],
                             gemm(1., Vt_T[t,:,:] - Vt_tm1[t,:,:], J[t-1,:,:],
                                  0., None, 0, 1),
                             0., None, 0, 0)
        
    # V_t_t1^T = Var(xt xt-1 | y1 ... yT)
    Vt_t1_T = np.zeros((T, dx, dx), 'd')
    #Vt_t1_T[T-1,:,:] = np.dot(np.eye(dx, dtype='d') - np.dot(Kt[T-1,:,:], C),
    #                        np.dot(A, Vt_t[T-2,:,:]))
    Vt_t1_T[T-1,:,:] = gemm(1., np.eye(dx, dtype='d') - np.dot(Kt[T-1,:,:], C),
                            gemm(1., A, Vt_t[T-2,:,:], 0., None, 0, 0),
                            0., None, 0, 0)
    for t in range(T-1, 1, -1):
        #Vt_t1_T[t-1,:,:] = np.dot(Vt_t[t-1,:,:], tr(J[t-2,:,:])) +\
        #                   np.dot(J[t-1,:,:],
        #                        np.dot(Vt_t1_T[t,:,:] - np.dot(A,Vt_t[t-1,:,:]),
        #                             tr(J[t-2,:,:])))
        Vt_t1_T[t-1,:,:] = gemm(1., Vt_t[t-1,:,:], J[t-2,:,:], 0., None, 0, 1) +\
                           gemm(1., J[t-1,:,:],
                                gemm(1., Vt_t1_T[t,:,:] - \
                                         gemm(1., A, Vt_t[t-1,:,:], 0., None, 0, 0),
                                     J[t-2,:,:], 0., None, 0, 1),
                                0., None, 0, 0)

    return xt_T, Vt_T, Vt_t1_T
