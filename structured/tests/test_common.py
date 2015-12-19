import unittest

import numpy as np
from numpy import diag, dot
from numpy.random import normal
from numpy.testing import assert_array_almost_equal

from structured import common


class TestCommon(unittest.TestCase):

    def test_mult_diag(self):
        dim = 20
        d = np.random.random(size=(dim,))
        dd = diag(d)
        mtx = np.random.random(size=(dim, dim))
    
        res1 = dot(dd, mtx)
        res2 = common.mult_diag(d, mtx, left=True)
        assert_array_almost_equal(res1, res2, 10)
        res1 = dot(mtx, dd)
        res2 = common.mult_diag(d, mtx, left=False)
        assert_array_almost_equal(res1, res2, 10)

    def test_bimult(self):
        dy, dc, dsc = 2, 7, 5
        for i in range(100):
            w = normal(0., 1., size=(dy, dc, dsc))
            c = normal(0., 1., size=(1, dc))
            s = normal(0., 1., size=(1, dc, dsc))
            # bimult
            y1 = common.bimult(w, c, s)
            # ad-hoc
            y2 = 0.
            for i in range(dc):
                y2 += dot(w[:,i,:], s[0,i,:])*c[0,i]
            assert_array_almost_equal(y1[0,:], y2, 7)
    
    def test_evaluate_style_subspace(self):
        dy, dc, dsc = 10, 3, 2
        w = np.random.random((dy,dc,dsc))
        w2 = w.copy()
        w2[:,0,0] = 0.
        w[:,2,1] = 0.
    
        vols, dimrel = common.evaluate_style_subspace(w, w2)
        assert_array_almost_equal(vols, [0., 0., 0.], 10)
        assert_array_almost_equal(dimrel, [1., 0., 2.], 10)
