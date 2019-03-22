import unittest, os
import numpy as np
import cherenkovdeconvolution.util as py_util

# import CherenkovDeconvolution.Util with the alias 'jl_util' from Julia
from julia.CherenkovDeconvolution import Util as CherenkovDeconvolution_Util
jl_util = CherenkovDeconvolution_Util # hack to achieve a lowercase alias unsupported by pyjulia..

@unittest.skipUnless(os.environ.get('TEST_JULIA')=='true', "Set TEST_JULIA=true to enable test")
class JlUtilTestSuite(unittest.TestCase):
    """Check the equivalence of cherenkovdeconvolution.util between Python and Julia."""
    
    @unittest.skip("Not yet implemented")
    def test_jl_fit_R(self):
        """Test the function cherenkovdeconvolution.util.fit_R."""
        pass # TODO test fit_R (required by IBU and RUN)
    
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

