import unittest
import numpy as np
import cherenkovdeconvolution.stepsize as stepsize


class StepsizeTestSuite(unittest.TestCase):
    """Test cases for the cherenkovdeconvolution.stepsize module."""
    
    
    def test_decay_mul(self):
        """Test the function cherenkovdeconvolution.stepsize.decay_mul."""
        # test on random arguments
        for i in range(10):
            num_bins = np.random.randint(1, 100)
            start    = np.random.uniform()
            eta      = np.random.uniform()
            
            alpha_fun = stepsize.decay_mul(eta, start)
            for k in range(1, 10):
                with self.subTest(i = i, k = k):
                    f_prev = np.random.uniform(num_bins)
                    pk     = np.random.uniform(num_bins)
                    self.assertEqual(alpha_fun(k, pk, f_prev), start * k**(eta-1))


    def test_decay_exp(self):
        """Test the function cherenkovdeconvolution.stepsize.decay_exp."""
        # test on random arguments
        for i in range(10):
            num_bins = np.random.randint(1, 100)
            start    = np.random.uniform()
            eta      = np.random.uniform()
            
            alpha_fun = stepsize.decay_exp(eta, start)
            for k in range(1, 10):
                with self.subTest(i = i, k = k):
                    f_prev = np.random.uniform(num_bins)
                    pk     = np.random.uniform(num_bins)
                    self.assertEqual(alpha_fun(k, pk, f_prev), start * eta**(k-1))

