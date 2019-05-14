import unittest, os
import numpy as np
import cherenkovdeconvolution as py_init

from pytest import importorskip

# import CherenkovDeconvolution with the alias 'jl_init' from Julia
jl_init = importorskip('julia.CherenkovDeconvolution')

class JlInitTestSuite(unittest.TestCase):
    """Check the equivalence of cherenkovdeconvolution.__init__ between Python and Julia."""
    
    def test_jl_recode(self):
        """Test the function cherenkovdeconvolution._recode_indices and _recode_result."""
        for i in range(10):
            with self.subTest(i = i):
                bins_total = range(np.random.randint(2, 100))
                bins = np.unique(np.random.randint(len(bins_total), size=len(bins_total)-1))
                num_samples = np.random.randint(1, 1000)
                x1 = np.random.choice(bins, size = num_samples)
                x2 = np.random.choice(bins, size = num_samples)
                py_dict, py_x1, py_x2 = py_init._recode_indices(bins_total, x1, x2)
                jl_dict, jl_x1, jl_x2 = jl_init._recode_indices(bins_total, x1, x2)
                np.testing.assert_array_equal(py_x1, jl_x1 - 1) # julia indices start at 1
                np.testing.assert_array_equal(py_x2, jl_x2 - 1)
                
                f1 = py_init.util.normalizepdf(np.bincount(x1, minlength=len(bins_total)))
                f2 = py_init.util.normalizepdf(np.bincount(x2, minlength=len(bins_total)))
                f1_r = py_init.util.normalizepdf(np.bincount(py_x1, minlength=len(bins)))
                f2_r = py_init.util.normalizepdf(np.bincount(py_x2, minlength=len(bins)))
                py_f1 = py_init._recode_result(f1_r, py_dict) # recode the result
                py_f2 = py_init._recode_result(f2_r, py_dict)
                np.testing.assert_array_equal(f1, py_f1)
                np.testing.assert_array_equal(f2, py_f2)
                
if __name__ == '__main__':
    unittest.main()
