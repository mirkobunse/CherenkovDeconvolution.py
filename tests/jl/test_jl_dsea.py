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
                
                # test Laplace correction
                w_bin[0] = 0
                py_w_train = py_dsea._dsea_weights(y_train, w_bin, normalize = False)
                jl_w_train = jl_dsea._dsea_weights(y_train+1, w_bin)
                np.testing.assert_allclose(py_w_train, jl_w_train) # test again
    
    @unittest.skip("Not yet implemented")
    def test_jl_train_and_predict_proba(self):
        """Test the function cherenkovdeconvolution.methods.dsea._train_and_predict_proba."""
        no_classifier = object() # empty object without fit and predict_proba methods
        X_data  = np.random.uniform(size = (4,3))
        X_train = np.random.uniform(size = (6,3))
        y_train = np.random.randint(2, size = 6)
        w_train = np.random.uniform(size = 6)
        with self.assertRaises(AttributeError): # no_classifier has no attribute 'fit'
            dsea._train_and_predict_proba(no_classifier, X_data, X_train, y_train, w_train)
    
    @unittest.skip("Not yet implemented")
    def test_jl_dsea_reconstruct(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_reconstruct."""
        for i in range(10):
            proba = np.random.uniform(size = (6,3))
            f = dsea._dsea_reconstruct(proba)
            self.assertAlmostEqual(np.sum(f), 1)
    
    @unittest.skip("Not yet implemented")
    def test_jl_dsea_step(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_step."""
        for i in range(10):
            num_bins = np.random.randint(1, 100)
            k_dummy  = np.random.randint(1, 100)
            f        = np.random.uniform(size = num_bins)
            f_prev   = np.random.uniform(size = num_bins)
            alpha_const = 2 * np.random.uniform(-1) # in [-2, 2)
            
            # test constant step size
            f_plus, alpha_out = dsea._dsea_step(k_dummy, f, f_prev, alpha_const)
            self.assertEqual(alpha_const, alpha_out)
            self.assertTrue(np.all( f_plus == f_prev + (f - f_prev) * alpha_const ))

if __name__ == '__main__':
    unittest.main()

