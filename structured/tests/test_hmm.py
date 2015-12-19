import unittest

import numpy as np
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_array_equal

from structured import hmm


class TestHMM(unittest.TestCase):
    
    def test_hmm_0(self):
        # random markov model, deterministic output
        nstates = 5
        T = 100
        pi = random(nstates)
        tr = random((nstates, nstates))
        def output_distr(y, i, t):
            return np.array(y==i+2, 'd')[0]
        # states
        x = random(T)
        x = np.floor(x*nstates)
        # observations
        y = np.transpose(np.array([x+2], 'd'))

        # compute epsilon for this case
        eps0 = np.zeros((T-1, nstates, nstates), 'd')
        for t in range(T-1):
            eps0[t, int(x[t]), int(x[t+1])] = 1.

        gamma, L, eps = hmm.forward_backward(
            y, pi, tr, output_distr, store_A=True)

        for t in range(T):
            x_t = int(y[t,0]-2)
            self.assertEqual(sum(gamma[t, :]), 1.0)
            self.assertEqual(gamma[t, x_t], 1.0)
        assert_array_almost_equal(eps, eps0, 7)

        gamma, L, eps = hmm.forward_backward(
            y, pi, tr, output_distr, store_A=False)

        for t in range(T):
            x_t = int(y[t,0]-2)
            self.assertEqual(sum(gamma[t, :]), 1.0)
            self.assertEqual(gamma[t, x_t], 1.0)
        assert_array_almost_equal(eps, eps0, 7)
    
    def test_hmm_1(self):
        # hand-verified markov model
        pi = np.array([0.2, 0.8])
        tr = np.array([[0.1, 0.9], [0.8, 0.2]])
        # observations
        obs = np.array([[0.5, 0.2, 0.3], [0.1, 0.8, 0.1]])
        def output_distr(y, i ,t):
            return obs[i, y[0]]
        y = np.array([[2],[0],[1]],  'i')
        # hand-computed probability of x_t being 1 and likelihood
        p1 = [0.85970149253731343,
              0.079601990049751242,
              0.93532338308457708]
        py = 0.02814

        gamma, L, epsilon = hmm.forward_backward(y, pi, tr, output_distr)
        # test that it returns a p.distr. for each time point
        not_one = max(abs(np.sum(gamma, axis=1)-1.))
        self.assertLess(not_one, 1e-7)
        # test that the distr. is the right one
        max_dist = max(abs(gamma[:,1]-p1))
        self.assertLess(max_dist, 1e-7)
        # test the value of the likelihood
        self.assertLess(abs(np.log(py)-L), 1e-7)

        self.assertEqual(sum(epsilon[0,:,:].ravel()), 1.0)
        self.assertEqual(sum(epsilon[1,:,:].ravel()), 1.0)
        
    def test_sample_chain(self):
        one = 1.-1e-5
        pi = [one, 0., 0.]
        ttr = np.zeros((3,3), dtype='d')
        ttr[0,1] = one
        ttr[1,2] = one
        ttr[2,0] = one
        x = hmm.sample_chain(6, 3, pi, ttr)
        for i in range(3):
            assert_array_equal(x[:,i], [0.,1.,2.,0.,1.,2.])
        
    def test_get_prior(self):
        # hand-verified markov model (no observations)
        pi = np.array([0.2, 0.8])
        ttr = np.array([[0.1, 0.9], [0.8, 0.2]])

        # hand-computed probability of x_t being 1
        p1 = [0.8, 0.34, 0.662]
        eps0 = [[[0.02, 0.18], [0.64, 0.16]], [[0.066, 0.594], [0.272, 0.068]]]

        p, L, eps = hmm.get_prior(3, pi, ttr)
        # test that it returns a p.distr. for each time point
        not_one = max(abs(np.sum(p, axis=1)-1.))
        self.assertLess(not_one, 1e-7)
        # test that the distr. is the right one
        assert_array_almost_equal(p1, p[:,1], 7)
        assert_array_almost_equal(eps, eps0, 7)
