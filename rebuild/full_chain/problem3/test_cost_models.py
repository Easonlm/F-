"""Unit tests for every quality cost model and FLOP unit conversion.

Run: python -m unittest discover -s problem3 -p 'test_*.py'
"""
import unittest
import numpy as np
from core import g,dg,total_cost,cost_gradient


class CostModelTests(unittest.TestCase):
    def check_kind(self,kind):
        n,d,q,q0,ctx=2.5,7.0,.8,.6,32768
        total,train,quality,attention=total_cost(n,d,q,q0,ctx,kind)
        self.assertAlmostEqual(train/(n*d*1e18),6)
        self.assertTrue(np.isclose(quality/(d*1e9),g(q,kind)-g(q0,kind),rtol=1e-14))
        self.assertAlmostEqual(attention/(n*d*1e18),2e-4*ctx)
        self.assertAlmostEqual(total,train+quality+attention,delta=total*1e-14)
        h=1e-6
        self.assertTrue(np.isclose(dg(q,kind),(g(q+h,kind)-g(q-h,kind))/(2*h),rtol=1e-7))
        grad=cost_gradient(n,d,q,q0,ctx,kind)
        numeric=[(total_cost(n+h,d,q,q0,ctx,kind)[0]-total_cost(n-h,d,q,q0,ctx,kind)[0])/(2*h),
                 (total_cost(n,d+h,q,q0,ctx,kind)[0]-total_cost(n,d-h,q,q0,ctx,kind)[0])/(2*h),
                 (total_cost(n,d,q+h,q0,ctx,kind)[0]-total_cost(n,d,q-h,q0,ctx,kind)[0])/(2*h)]
        self.assertTrue(np.allclose(grad,numeric,rtol=1e-7))

    def test_exponential(self):self.check_kind('exponential')
    def test_power(self):self.check_kind('power')
    def test_logarithmic(self):self.check_kind('logarithmic')
    def test_context_critical(self):
        _,train,_,att=total_cost(1,1,.6,.6,30000,'exponential')
        self.assertAlmostEqual(train,att)


if __name__=='__main__':unittest.main()
