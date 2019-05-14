import unittest, os
import numpy as np
import cherenkovdeconvolution.util as util
import cherenkovdeconvolution.discretize as discretize
import cherenkovdeconvolution.methods.run as py_run
from pytest import importorskip

# import CherenkovDeconvolution with the alias 'jl_run' from Julia

jl_run = importorskip('julia.CherenkovDeconvolution')

class JlDseaTestSuite(unittest.TestCase):
    """Check the equivalence of DSEA between Python and Julia."""
    
    @unittest.skip('Not yet implemented')
    def test_jl_run_maxl(self):
        """Test the functions cherenkovdeconvolution.methods.run._maxl_*."""
        for i in range(10):
            with self.subTest(i = i):
                I = np.random.randint(1, 100)
                J = np.random.randint(1, 1000)
                R = np.random.rand(J, I)
                g = np.random.rand(J)
                # py_B = py_ibu._ibu_reverse_transfer(R, f_0)
                # jl_B = jl_ibu._ibu_reverse_transfer(R, f_0)
                # np.testing.assert_allclose(py_B, jl_B)
    
    def test_jl_run(self):
        """Test complete deconvolution runs with RUN."""
        from sklearn.datasets import load_iris
        iris = load_iris()
        
        # discretize the observed quantity into up to 6 clusters
        y_iris = iris.target # already discrete
        bins_y = np.sort(np.unique(y_iris))
        x_iris = discretize.TreeDiscretizer(iris.data, y_iris, 6).discretize(iris.data)
        
        for i in range(10):
            p_iris = np.random.permutation(len(iris.target))
            x_data  = x_iris[p_iris[0:50]]
            x_train = x_iris[p_iris[50:150]]
            y_train = y_iris[p_iris[50:150]]
            py_f = py_run.deconvolve_evt(x_data, x_train, y_train, bins_y, K=2) # single iteration
            jl_f = jl_run.run(x_data, x_train, y_train+1, bins_y+1, K=2)
            np.testing.assert_allclose(py_f, jl_f)
            
            py_f = py_run.deconvolve_evt(x_data, x_train, y_train, bins_y, K=100) # K = 100
            jl_f = jl_run.run(x_data, x_train, y_train+1, bins_y+1, K=100)
            np.testing.assert_allclose(py_f, jl_f, rtol=1e-4) # higher tolerance needed

if __name__ == '__main__':
    unittest.main()

