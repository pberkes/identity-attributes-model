import unittest

import mdp
import numpy as np
from numpy import dot
from numpy.testing import assert_allclose, assert_array_almost_equal

from structured import kalman


def random_contraction(dim, dtype='d'):
    """Return a random rotation matrix."""
    mtx = np.random.random((dim, dim))
    mtx = mtx + np.transpose(mtx)
    d, A = mdp.utils.symeig(mtx, overwrite=1)
    d = d/np.amax(abs(d))*0.9
    mtx = dot(A, dot(np.diag(d), A.T))
    return mtx.astype(dtype)


def random_LDS(dy, dx, Qstd=0.1, V1std=0.1, Rstd=0.1):
    # mixing matrix
    A = random_contraction(dx)

    # eigenvalues of the covariance matrix Q
    sigma_Q = np.random.random((dx,))*Qstd + 0.05
    Q = mdp.utils.symrand(sigma_Q)

    # initial distr
    pi1 = np.random.normal(0., 1., size=(dx,))
    sigma_V1 = np.random.random((dx,))*V1std + 0.01
    V1 = mdp.utils.symrand(sigma_V1)

    # obs matrix
    C = np.random.normal(0., 1., size=(dy, dx))

    # eigenvalues of the covariance matrix R
    sigma_R = np.random.random((dy,))*Rstd
    R = mdp.utils.symrand(sigma_R)

    return A, C, pi1, V1, Q, R


class TestKalman(unittest.TestCase):

    def test_sample_LDS(self):
        dx = 2
        A, C, pi1, V1, Q, R = random_LDS(1, dx, Qstd=0.3)
        #
        T = 5
        cov_mtx = [mdp.utils.CovarianceMatrix() for t in range(T)]
        for n in range(5000):
            x = kalman.sample_LDS(5, A, pi1, V1, Q)
            for t in range(T):
                if t==0: cov_mtx[0].update(x[0:1,:])
                else: cov_mtx[t].update(x[t:t+1] - dot(x[t-1:t], A.T))
        # fix
        cov, avg = [None]*T, [None]*T
        for t in range(T):
            cov[t], avg[t], tmp = cov_mtx[t].fix()
        # verify for t==0
        assert_allclose(avg[0], pi1, atol=0.1)
        assert_allclose(cov[0], V1, atol=0.1)
        # verify for t>0
        zero = np.zeros((dx,), 'd')
        for t in range(1, T):
            assert_array_almost_equal(avg[t], zero, 1)
            assert_array_almost_equal(cov[t], Q, 2)

    def test_smoother(self):
        T, dy, dx = 100, 3, 2
        A, C, pi1, V1, Q, R = random_LDS(dy, dx, Rstd=0.0001)
        x = kalman.sample_LDS(T, A, pi1, V1, Q)
        y = kalman.execute_LDS(x, C, R=None)
        #
        xmn, xcov, Vt_t1_T = kalman.smooth(y, A, C, pi1, V1, Q, R)
        assert_allclose(xmn, x, atol=0.01)

        # non-stationary noise
        T, dy, dx = 100, 2, 2
        A = np.eye(dx, dtype='d')
        Q = np.eye(dx, dtype='d')*1e5
        C = np.eye(dx, dtype='d')
        pi1 = np.ones((dx,), dtype='d')
        V1 = np.eye(dy, dtype='d')*1e5
        R = np.zeros((T, dy, dy), dtype='d')
        sgm1, sgm2 = 1e-2, 5e-2
        for t in range(T/2):
            R[t,:,:] = np.eye(dy, dtype='d')*1e-2
        for t in range(T/2, T):
            R[t,:,:] = np.eye(dy, dtype='d')*5e-2
        y = np.ones((T,2), dtype='d')

        xmn, xcov, Vt_t1_T = kalman.smooth(y, A, C, pi1, V1, Q, R)

        self.assertLess(abs(xcov[20,0,0]-sgm1), 1e-7)
        self.assertLess(abs(xcov[80,0,0]-sgm2), 1e-7)
