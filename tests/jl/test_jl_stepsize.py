import unittest, os
import numpy as np
import cherenkovdeconvolution.stepsize as py_stepsize

# import CherenkovDeconvolution with the alias 'jl_stepsize' from Julia
from julia import CherenkovDeconvolution
jl_stepsize = CherenkovDeconvolution # hack to achieve a lowercase alias unsupported by pyjulia..

@unittest.skipUnless(os.environ.get('TEST_JULIA')=='true', "Set TEST_JULIA=true to enable test")
class JlStepsizeTestSuite(unittest.TestCase):
    """Check the equivalence of cherenkovdeconvolution.stepsize between Python and Julia."""
    
    def test_jl_decay_mul(self):
        """Test the function cherenkovdeconvolution.stepsize.decay_mul."""
        for i in range(10):
            num_bins = np.random.randint(1, 100)
            start    = np.random.uniform()
            eta      = np.random.uniform()
            
            py_alpha_fun = py_stepsize.decay_mul(eta, start)
            jl_alpha_fun = jl_stepsize.alpha_decay_mul(eta, start)
            for k in range(1, 10):
                with self.subTest(i = i, k = k):
                    f_prev = np.random.rand(num_bins)
                    pk     = np.random.rand(num_bins)
                    self.assertAlmostEqual(
                      py_alpha_fun(k, pk, f_prev),
                      jl_alpha_fun(k, pk, f_prev)
                    )

    def test_jl_decay_exp(self):
        """Test the function cherenkovdeconvolution.stepsize.decay_exp."""
        for i in range(10):
            num_bins = np.random.randint(1, 100)
            start    = np.random.uniform()
            eta      = np.random.uniform()
            
            py_alpha_fun = py_stepsize.decay_exp(eta, start)
            jl_alpha_fun = jl_stepsize.alpha_decay_exp(eta, start)
            for k in range(1, 10):
                with self.subTest(i = i, k = k):
                    f_prev = np.random.rand(num_bins)
                    pk     = np.random.rand(num_bins)
                    self.assertAlmostEqual(
                      py_alpha_fun(k, pk, f_prev),
                      jl_alpha_fun(k, pk, f_prev)
                    )

