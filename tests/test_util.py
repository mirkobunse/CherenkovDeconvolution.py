# from .context import cherenkovdeconvolution
import unittest
import numpy as np
import cherenkovdeconvolution.util as util


class UtilTestSuite(unittest.TestCase):
    """Test cases for the cherenkovdeconvolution.util module."""
    
    
    def test_equidistant_bin_edges(self):
        """Test the function cherenkovdeconvolution.util.equidistant_bin_edges."""
        # test on random arguments
        for i in range(10):
            with self.subTest(i=i):
                minimum  = np.random.uniform(-1)      # in [-1, 1)
                maximum  = np.random.uniform(minimum) # in [minimum, 1)
                num_bins = np.random.randint(1, 1000)
                bin_edges = util.equidistant_bin_edges(minimum, maximum, num_bins)
                self.assertEqual(len(bin_edges), num_bins + 1)
                self.assertEqual(bin_edges.min(), minimum)
                self.assertEqual(bin_edges.max(), maximum)
        
        # test exceptions
        minimum  = np.random.uniform(-1)
        maximum  = np.random.uniform(minimum)
        num_bins = np.random.randint(1, 1000)
        with self.assertRaises(ValueError):
            util.equidistant_bin_edges(minimum, maximum, 0) # num_bins has to be > 1
        with self.assertRaises(ValueError):
            util.equidistant_bin_edges(maximum, minimum, num_bins) # max > min is required
        with self.assertRaises(ValueError):
            util.equidistant_bin_edges(minimum, minimum, num_bins) # max > min is required (equality)
    
    
    @unittest.skip("Not yet implemented")
    def test_fit_R(self):
        """Test the function cherenkovdeconvolution.util.fit_R."""
        pass # TODO
    
    
    def test_normalizepdf(self):
        """Test the function cherenkovdeconvolution.util.normalizepdf."""
        # test on random arguments
        for i in range(10):
            with self.subTest(i=i):
                num_bins = np.random.randint(1, 1000)
                arr      = np.random.uniform(size = num_bins)
                narr     = util.normalizepdf(arr)
                self.assertTrue(np.any(arr != narr)) # not performed in place
                self.assertEqual(len(narr), num_bins)
                self.assertAlmostEqual(sum(narr), 1) # total equality violated by rounding
                util.normalizepdf(arr, copy = False)
                self.assertTrue(np.all(arr == narr)) # in place version
        
        # test exceptions
        intarr = np.random.randint(1000, size = 10) # integer array
        self.assertAlmostEqual(sum(util.normalizepdf(intarr)), 1)
        with self.assertRaises(ValueError):
            util.normalizepdf(intarr, copy = False) # in place only allowed on floats
    
    
    @unittest.skip("Not yet implemented")
    def test_smooth_polynomial(self):
        """Test the function cherenkovdeconvolution.util.smooth_polynomial."""
        pass # TODO
    
    
    @unittest.skip("Not yet implemented")
    def test_chi2s(self):
        """Test the function cherenkovdeconvolution.util.chi2s."""
        pass # TODO
    
    
if __name__ == '__main__':
    unittest.main()
