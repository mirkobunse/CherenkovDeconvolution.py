import unittest, os
import numpy as np
import cherenkovdeconvolution.util as py_util

# import CherenkovDeconvolution.Util with the alias 'jl_util' from Julia
from julia.CherenkovDeconvolution import Util as CherenkovDeconvolution_Util
jl_util = CherenkovDeconvolution_Util # hack to achieve a lowercase alias unsupported by pyjulia..

@unittest.skipUnless(os.environ.get('TEST_JULIA')=='true', "Set TEST_JULIA=true to enable test")
class JlUtilTestSuite(unittest.TestCase):
    """Check the equivalence of cherenkovdeconvolution.util between Python and Julia."""
    
    def test_jl_fit_R(self):
        """Test the function cherenkovdeconvolution.util.fit_R."""
        for i in range(10):
            with self.subTest(i = i):
                num_samples = np.random.randint(1, 1000)
                bins_y  = range(np.random.randint(1, 100))
                bins_x  = range(np.random.randint(1, 100))
                y = np.random.randint(len(bins_y), size = num_samples) + 1
                x = np.random.randint(len(bins_x), size = num_samples) + 1
                py_R = py_util.fit_R(y, x, bins_y = bins_y, bins_x = bins_x)
                jl_R = jl_util.fit_R(y, x, bins_y = bins_y, bins_x = bins_x)
                np.testing.assert_allclose(py_R, jl_R)
                
                py_R = py_util.fit_R(y, x, bins_y = bins_y, bins_x = bins_x, normalize=False)
                jl_R = jl_util.fit_R(y, x, bins_y = bins_y, bins_x = bins_x, normalize=False)
                np.testing.assert_allclose(py_R, jl_R)
                
                py_R = py_util.fit_R(y, x) # no bins specified
                jl_R = jl_util.fit_R(y, x)
                np.testing.assert_allclose(py_R, jl_R, err_msg='I={}/{}, J={}/{}'.format(
                  len(np.unique(y)),
                  len(bins_y),
                  len(np.unique(x)),
                  len(bins_x)
                ))
    
    def test_jl_normalizepdf(self):
        """Test the function cherenkovdeconvolution.util.normalizepdf."""
        for i in range(10):
            with self.subTest(i = i):
                num_bins = np.random.randint(1, 1000)
                arr      = np.random.uniform(size = num_bins)
                py_narr = py_util.normalizepdf(arr)
                jl_narr = jl_util.normalizepdf(arr)
                np.testing.assert_allclose(py_narr, jl_narr)
    
    def test_jl_chi2s(self):
        """Test the function cherenkovdeconvolution.util.chi2s."""
        for i in range(10):
            with self.subTest(i = i):
                num_bins = np.random.randint(1, 1000)
                a        = np.random.randint(1000, size = num_bins)
                b        = np.random.randint(1000, size = num_bins)
                py_chi2s = py_util.chi2s(a, b)
                jl_chi2s = jl_util.chi2s(a, b)
                self.assertAlmostEqual(py_chi2s, jl_chi2s)
    
if __name__ == '__main__':
    unittest.main()

