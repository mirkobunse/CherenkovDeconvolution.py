import unittest, os
import numpy as np
import cherenkovdeconvolution.discretize as discretize
import cherenkovdeconvolution.util as util
import cherenkovdeconvolution.stepsize as py_stepsize

# import CherenkovDeconvolution with the alias 'jl_stepsize' from Julia
from julia import CherenkovDeconvolution
jl_stepsize = CherenkovDeconvolution # hack to achieve a lowercase alias unsupported by pyjulia..

@unittest.skipUnless(os.environ.get('TEST_JULIA')=='true', "Set TEST_JULIA=true to enable test")
class JlStepsizeTestSuite(unittest.TestCase):
    """Check the equivalence of cherenkovdeconvolution.stepsize between Python and Julia."""
    
    def test_jl_decay_mul(self):
        """Test the function cherenkovdeconvolution.stepsize.decay_mul."""
        for i in range(10):
            num_bins = np.random.randint(1, 100)
            start    = np.random.uniform()
            eta      = np.random.uniform()
            
            py_alpha_fun = py_stepsize.decay_mul(eta, start)
            jl_alpha_fun = jl_stepsize.alpha_decay_mul(eta, start)
            for k in range(1, 10):
                with self.subTest(i = i, k = k):
                    f_prev = np.random.rand(num_bins)
                    pk     = np.random.rand(num_bins)
                    self.assertAlmostEqual(
                      py_alpha_fun(k, pk, f_prev),
                      jl_alpha_fun(k, pk, f_prev)
                    )

    def test_jl_decay_exp(self):
        """Test the function cherenkovdeconvolution.stepsize.decay_exp."""
        for i in range(10):
            num_bins = np.random.randint(1, 100)
            start    = np.random.uniform()
            eta      = np.random.uniform()
            
            py_alpha_fun = py_stepsize.decay_exp(eta, start)
            jl_alpha_fun = jl_stepsize.alpha_decay_exp(eta, start)
            for k in range(1, 10):
                with self.subTest(i = i, k = k):
                    f_prev = np.random.rand(num_bins)
                    pk     = np.random.rand(num_bins)
                    self.assertAlmostEqual(
                      py_alpha_fun(k, pk, f_prev),
                      jl_alpha_fun(k, pk, f_prev)
                    )

    
    def test_jl_alpha_adaptive_run(self):
        """Test the function cherenkovdeconvolution.stepsize.alpha_adaptive_run."""
        from sklearn.datasets import load_iris
        iris = load_iris()
        
        # discretize the observed quantity into up to 6 clusters
        y_iris = iris.target # already discrete
        bins_y = np.sort(np.unique(y_iris))
        x_iris = discretize.TreeDiscretizer(iris.data, y_iris, 6).discretize(iris.data)
        
        for i in range(10):
            p_iris = np.random.permutation(len(iris.target))
            x_data  = x_iris[p_iris[0:50]]
            y_data  = y_iris[p_iris[0:50]]
            x_train = x_iris[p_iris[50:150]]
            y_train = y_iris[p_iris[50:150]]
            
            # find some random f_prev and f_next
            f_true = util.fit_pdf(y_data, bins_y)
            f_next = util.normalizepdf(2 * np.random.rand(len(f_true)) * f_true)
            f_prev = util.normalizepdf(2 * np.random.rand(len(f_true)) * f_true)
            pk = f_next - f_prev
            
            # optimize the step size
            py_a = py_stepsize.alpha_adaptive_run(x_data, x_train, y_train, 0, bins_y)
            jl_a = jl_stepsize.alpha_adaptive_run(x_data+1, x_train+1, y_train+1, 0, bins_y+1)
            self.assertAlmostEqual(py_a, jl_a)

if __name__ == '__main__':
    unittest.main()

