import unittest, os
import numpy as np
import cherenkovdeconvolution.util as util
import cherenkovdeconvolution.methods.dsea as py_dsea

# import CherenkovDeconvolution with the alias 'jl_dsea' from Julia
from julia import CherenkovDeconvolution
jl_dsea = CherenkovDeconvolution # hack to achieve a lowercase alias unsupported by pyjulia..

@unittest.skipUnless(os.environ.get('TEST_JULIA')=='true', "Set TEST_JULIA=true to enable test")
class JlDseaTestSuite(unittest.TestCase):
    """Check the equivalence of DSEA between Python and Julia."""
    
    def test_jl_dsea_weights(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_weights."""
        for i in range(10):
            with self.subTest(i = i):
                n_samples = np.random.randint(1, 1000)
                num_bins  = np.random.randint(1, 100)
                y_train   = np.random.randint(num_bins, size = n_samples)
                w_bin     = np.random.rand(num_bins)
                py_w_train = py_dsea._dsea_weights(y_train, w_bin, normalize = False)
                jl_w_train = jl_dsea._dsea_weights(y_train+1, w_bin) # julia indices start at 1
                np.testing.assert_allclose(py_w_train, jl_w_train)
                
                w_bin[0] = 0 # second run with Laplace correction
                py_w_train = py_dsea._dsea_weights(y_train, w_bin, normalize = False)
                jl_w_train = jl_dsea._dsea_weights(y_train+1, w_bin)
                np.testing.assert_allclose(py_w_train, jl_w_train) # test again
    
    def test_jl_dsea_reconstruct(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_reconstruct."""
        for i in range(10):
            proba = np.random.uniform(size = (6,3))
            py_f = py_dsea._dsea_reconstruct(proba)
            jl_f = jl_dsea._dsea_reconstruct(proba)
            np.testing.assert_allclose(py_f, jl_f)
    
    def test_jl_dsea_step(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_step."""
        for i in range(10):
            num_bins = np.random.randint(1, 100)
            k_dummy  = np.random.randint(1, 100)
            f        = np.random.uniform(size = num_bins)
            f_prev   = np.random.uniform(size = num_bins)
            alpha_const = 2 * np.random.uniform(-1) # in [-2, 2)
            py_f, py_alpha = py_dsea._dsea_step(k_dummy, f, f_prev, alpha_const)
            jl_f, jl_alpha = jl_dsea._dsea_step(k_dummy, f, f_prev, alpha_const)
            self.assertAlmostEqual(py_alpha, jl_alpha)
            np.testing.assert_allclose(py_f, jl_f)

if __name__ == '__main__':
    unittest.main()

