import unittest
import numpy as np
import cherenkovdeconvolution.methods.dsea as py_dsea

from pytest import importorskip



# import CherenkovDeconvolution with the alias 'jl_dsea' from Julia

jl_dsea = importorskip('julia.CherenkovDeconvolution')
jl_skl = importorskip('julia.CherenkovDeconvolution.Sklearn')


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

                w_bin[0] = 0 # second run with Laplace correction
                py_w_train = py_dsea._dsea_weights(y_train, w_bin, normalize = False)
                jl_w_train = jl_dsea._dsea_weights(y_train+1, w_bin)
                np.testing.assert_allclose(py_w_train, jl_w_train) # test again

    def test_jl_dsea_reconstruct(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_reconstruct."""
        for i in range(10):
            proba = np.random.uniform(size = (6,3))
            py_f = py_dsea._dsea_reconstruct(proba)
            jl_f = jl_dsea._dsea_reconstruct(proba)
            np.testing.assert_allclose(py_f, jl_f)

    def test_jl_dsea_step(self):
        """Test the function cherenkovdeconvolution.methods.dsea._dsea_step."""
        for i in range(10):
            num_bins = np.random.randint(1, 100)
            k_dummy  = np.random.randint(1, 100)
            f        = np.random.uniform(size = num_bins)
            f_prev   = np.random.uniform(size = num_bins)
            alpha_const = 2 * np.random.uniform(-1) # in [-2, 2)
            py_f, py_alpha = py_dsea._dsea_step(k_dummy, f, f_prev, alpha_const)
            jl_f, jl_alpha = jl_dsea._dsea_step(k_dummy, f, f_prev, alpha_const)
            self.assertAlmostEqual(py_alpha, jl_alpha)
            np.testing.assert_allclose(py_f, jl_f)

    def test_jl_dsea(self):
        """Test complete deconvolution runs with DSEA."""
        from sklearn.datasets import load_iris
        from sklearn.naive_bayes import GaussianNB
        iris = load_iris()
        for i in range(10):
            p_iris = np.random.permutation(len(iris.target))
            X_data  = iris.data[p_iris[0:50], :]
            X_train = iris.data[p_iris[50:150], :]
            y_train = iris.target[p_iris[50:150]]
            tp = jl_skl.train_and_predict_proba(GaussianNB())
            py_f = py_dsea.deconvolve(X_data, X_train, y_train, GaussianNB())
            jl_f = jl_dsea.dsea(X_data, X_train, y_train+1, tp)
            np.testing.assert_allclose(py_f, jl_f)
            py_f = py_dsea.deconvolve(X_data, X_train, y_train, GaussianNB(), K=10) # 10 iterations
            jl_f = jl_dsea.dsea(X_data, X_train, y_train+1, tp, K=10)
            np.testing.assert_allclose(py_f, jl_f)

if __name__ == '__main__':
    unittest.main()

