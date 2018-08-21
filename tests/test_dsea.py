import unittest
import numpy as np
import cherenkovdeconvolution.util as util
import cherenkovdeconvolution.methods.dsea as dsea


class DseaTestSuite(unittest.TestCase):
    """Test cases for the cherenkovdeconvolution.methods.dsea module."""
    
    
    def test_dsea_weights(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_weights."""
        # test on random arguments
        for i in range(10):
            with self.subTest(i = i):
                n_samples = np.random.randint(1, 1000)
                num_bins  = np.random.randint(1, 100)
                y_train   = np.random.randint(num_bins, size = n_samples)
                w_bin     = np.random.uniform(size = num_bins)
                w_train = dsea._dsea_weights(y_train, w_bin, normalize = False)
                
                # consider Laplace correction (correct weights pass the test without assertion)
                unequal = w_train != w_bin[y_train] # indices which have to be checked
                self.assertTrue(np.all(w_train[unequal] == 1/len(y_train)))
                self.assertTrue(np.all(w_bin[y_train[unequal]] <= 1/len(y_train)))
    
    
    def test_train_and_predict_proba(self):
        """Test the function cherenkovdeconvolution.methods.dsea._train_and_predict_proba."""
        # check exceptions (everything else depends on classifier object)
        no_classifier = object() # empty object without fit and predict_proba methods
        X_data  = np.random.uniform(size = (4,3))
        X_train = np.random.uniform(size = (6,3))
        y_train = np.random.randint(2, size = 6)
        w_train = np.random.uniform(size = 6)
        with self.assertRaises(AttributeError): # no_classifier has no attribute 'fit'
            dsea._train_and_predict_proba(no_classifier, X_data, X_train, y_train, w_train)
    
    
    def test_dsea_reconstruct(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_reconstruct."""
        # test on random arguments
        for i in range(10):
            proba = np.random.uniform(size = (6,3))
            f = dsea._dsea_reconstruct(proba)
            self.assertAlmostEqual(np.sum(f), 1)
    
    
    def test_dsea_step(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_step."""
        # test on random arguments
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
        
            # test exceptions
            def no_alpha(k): return 1/k # wrong number of arguments
            with self.assertRaises(TypeError):
                dsea._dsea_step(k_dummy, f, f_prev, no_alpha)
    
    
if __name__ == '__main__':
    unittest.main()

