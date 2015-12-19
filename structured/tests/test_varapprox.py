import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from structured import common, gibbs, objects, varapprox
from structured.common import mult_diag


def _random_model(T, dy, dc, dsc, params='noninformative', w_real=None):
    # params: 'noninformative', 'exact',
    #          dictionary with tT, p00, p11, tlbd, lbds
    
    # w_mean set to the correct weights
    if w_real is None:
        w_real = common.random_weights_orth(dy, dc, dsc)
    T_real = np.array([[0.8,0.2],[0.1,0.9]], dtype='d')
    lbd_real = np.linspace(0.8, 0.2, dsc)

    if params=='noninformative':
        tT, p00, p11 = 1., 0.5, 0.5
        tlbd, lbds = 1., np.linspace(0.0,0.0,dsc)
        tnoise, noise_prec = 1., 1./0.01
        gamma = 1.
    elif params=='exact':
        tT, p00, p11 = 1e3, T_real[0,0], T_real[1,1]
        tlbd, lbds = 1e3, lbd_real.copy()
        tnoise, noise_prec = 1e5, 1e10
        gamma = 1.e-5

    w_mean = w_real.copy()
    w_var = np.zeros((dy,dc,dsc), dtype='d') + 1e-10
    params = objects.Parameters(dy, dc, dsc, 0.5, tT, p00, p11,
                                tlbd, lbds, tnoise, noise_prec,
                                gamma=gamma,
                                w_mean=w_mean, w_var=w_var,
                                w_real=w_real,
                                T_real = T_real, lbd_real=lbd_real)
    y, c, s = params.sample_all(T, noise=False, s_off=True)
    state = objects.State(y, params)
    state.set_to_prior()

    return params, state, c, s


class TestVarapprox(unittest.TestCase):
    def test_e_step(self):
        params, state, c, s = _random_model(100, 18, 5, 3, params='exact')
    
        varapprox.e_step(state, params)
        
        assert_array_almost_equal(c, state.pc, 7)
        # s_mn is zero if content off
        for i in range(params.dc):
            state.s_mn[:,i,:] *= np.repeat(c[:,i,np.newaxis], (params.dsc,), 1)
        assert_array_almost_equal(s, state.s_mn, 7)
        return params, state, c, s
    
    def test_m_step(self):
        T, dy, dc, dsc = 1000, 14, 4, 3
        params, state, c, s = _random_model(T, dy, dc, dsc, params='exact')
    
        params.w_mean = common.random_weights_orth(dy, dc, dsc)
        params.w_var = np.zeros((dy,dc,dsc), dtype='d') + 1e-1
        params.init_wDw()
    
        # vars from the data
        state.set_to(c,s)
        varapprox.close_e_step(params, state)
        
        # learn w
        varapprox.m_step(state, params, max_iter=1000)
        # evaluate
        assert_array_almost_equal(params.w_real, params.w_mean, 8)
    
    def test_m_step_rho(self):
        T, dy, dc, dsc = 5000, 14, 3, 2
        params, state, c, s = _random_model(T, dy, dc, dsc, params='exact')
    
        tnoise, noise_prec = 1., 25.
        params.init_noise_prior(tnoise, noise_prec)
    
        # add structured noise to the observations
        noise_std = np.linspace(1., 0.01, dy)*0.2*state.y.std()
        state.y += mult_diag(noise_std, np.random.normal(size=(T, dy)), left=False)
        # vars from the data
        state.set_to(c,s)
        varapprox.close_e_step(params, state)
        
        # learn w
        varapprox.m_step(state, params, max_iter=1000)
        # evaluate
        assert_array_almost_equal(np.sqrt(1./params.rho_mn), noise_std, 2)
        return params, state, c, s
    
    def test_perturbation(self):
        T, dy, dc, dsc = 2000, 14, 3, 2
        params, state, c, s = _random_model(T, dy, dc, dsc)
    
        params.w_mean += np.random.normal(scale=1e-1, size=(dy,dc,dsc))
        params.w_var = np.zeros((dy,dc,dsc), dtype='d') + 1e-3
        params.init_wDw()
    
        old_free = -np.inf
        for i in range(10):
            print 'iteration %d/10' % i
            for j in range(3):
                varapprox.e_step(state, params)
                varapprox.close_e_step(params, state)
                new_free = varapprox.free_energy(params, state)
                self.assertGreaterEqual(new_free-old_free, 0.0)
                old_free = new_free
    
            varapprox.close_e_step(params, state)
            varapprox.m_step(state, params, max_iter=2000)
            new_free = varapprox.free_energy(params, state)
            self.assertGreaterEqual(new_free-old_free, 0.0)
                
            vols, dimrel, bestc = common.evaluate_style_subspace_perms(params.w_real, params.w_mean)
            print vols
            
        # evaluate
        vols, dimrel, bestc = common.evaluate_style_subspace_perms(params.w_real, params.w_mean)
        self.assertLess(max(vols), 1e-2)
        assert_array_almost_equal(params.lbd_mn.mean(axis=0), params.lbd_real[0,:], 1)
        assert_array_almost_equal(params.T_mn, params.T_real, 1)

    def test_learn_gamma(self):
        dy, dc, dsc = 7, 2, 3
        w_real = common.random_weights_orth(dy, dc, dsc) * np.sqrt(dy)
        w_real[:,0,0] = 0.
        w_real[:,1,0] *= np.sqrt(2)
        params, state, c, s = _random_model(1000, dy, dc, dsc, params='exact',
                                            w_real=w_real)
    
        # vars from the data
        state.set_to(c,s)
    
        varapprox.learn_gamma(params)
        self.assertGreater(params.gamma[0,0], 1e4)
        assert_almost_equal(params.gamma[1,0], 0.5, 8)
        for i in range(dc):
            for j in range(1,dsc):
                assert_almost_equal(params.gamma[i,j], 1., 8)
        
    def test_sample_c(self):
        params, state, c, s = _random_model(50, 10, 2, 3, params='exact')
        y, c, s = params.sample_all(50, noise=False, s_off=False)
        state = objects.State(y, params)
        state.set_to_prior()
    
        nsamples = 300
        samples = np.zeros((nsamples+1, 50, 2))
        samples[0,:] = state.pc.copy()
        for i in range(nsamples):
            samples[i+1,:] = gibbs.sample_c(state.y, samples[i,:], s, params)
    
        assert_almost_equal(c, samples.mean(axis=0), 2)
    
    def test_sample_s(self):
        T, dy, dc, dsc = 50, 10, 3 ,2
        params, state, c, s = _random_model(T, dy, dc, dsc, params='noninformative')
    
        nsamples = 300
        s_samples = np.zeros((nsamples+1, T, dc, dsc))
        s_samples[0,:] = state.s_mn.copy()
        for i in range(nsamples):
            s_samples[i+1,:] = gibbs.sample_s(state.y, c, s_samples[i,:], params)
    
        assert_almost_equal(s, s_samples.mean(axis=0), 1)
