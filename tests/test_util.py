import unittest
import numpy as np
import cherenkovdeconvolution.util as util


class UtilTestSuite(unittest.TestCase):
    """Test cases for the cherenkovdeconvolution.util module."""
    
    
    def test_equidistant_bin_edges(self):
        """Test the function cherenkovdeconvolution.util.equidistant_bin_edges."""
        # test on random arguments
        for i in range(10):
            with self.subTest(i = i):
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
        pass # TODO test fit_R (required by IBU and RUN)
    
    
    def test_normalizepdf(self):
        """Test the function cherenkovdeconvolution.util.normalizepdf."""
        # test on random arguments
        for i in range(10):
            with self.subTest(i = i):
                num_bins = np.random.randint(1, 1000)
                arr      = np.random.uniform(size = num_bins)
                narr = util.normalizepdf(arr)
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
    
    
    def test_smooth_polynomial(self):
        """Test the function cherenkovdeconvolution.util.smooth_polynomial."""
        # test on random arguments
        for i in range(10):
            num_bins = np.random.randint(100, 1000)
            arr      = np.random.uniform(size = num_bins)
            
            # simple order 1 check
            with self.subTest(i = i):
                sarr = util.smooth_polynomial(arr, order = 1)
                diffs = sarr[1:] - sarr[:-1] # array of finite differences
                mdiff = np.mean(diffs)       # mean difference
                self.assertTrue(np.allclose(diffs, mdiff)) # all differences approx. equal
                
            # multiple smoothings return approximately same array
            order = i + 1
            with self.subTest(order = order):
                sarr1 = util.smooth_polynomial(arr,   order = order)
                sarr2 = util.smooth_polynomial(sarr1, order = order)
                self.assertTrue(np.allclose(sarr1, sarr2))
        
        # test exceptions
        with self.assertRaises(ValueError):
            util.smooth_polynomial(np.random.uniform(size = 3), order = 3) # order < len(arr)
    
    
    def test_chi2s(self):
        """Test the function cherenkovdeconvolution.util.chi2s."""
        # test on random arguments
        for i in range(10):
            with self.subTest(i = i):
                num_bins = np.random.randint(1, 1000)
                a        = np.random.randint(1000, size = num_bins)
                b        = np.random.randint(1000, size = num_bins)
                chi2s = util.chi2s(a, b)
                self.assertGreaterEqual(chi2s, 0)
                self.assertEqual(util.chi2s(util.normalizepdf(a),
                                            util.normalizepdf(b),
                                            normalize = False),  chi2s)
        
        # test increase on diverging arrays
        num_bins = np.random.randint(2, 1000)
        a = np.zeros(num_bins)
        b = np.ones(num_bins)
        a[1] = 1
        last_chi2s = util.chi2s(a, b)
        for i in range(10):
            b[2] += 1 - np.random.uniform() # in (0, 1]
            with self.subTest(b2 = b[2]):
                chi2s = util.chi2s(a, b)
                self.assertGreater(chi2s, last_chi2s)
                last_chi2s = chi2s
        
        # test exceptions
        with self.assertRaises(ValueError):
            util.chi2s(np.random.uniform(size = 3), np.random.uniform(size = 4))
    
    
if __name__ == '__main__':
    unittest.main()

