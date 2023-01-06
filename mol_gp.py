import gpflow
from gpflow.mean_functions import Constant
from sklearn.preprocessing import StandardScaler
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
import tensorflow as tf
import numpy as np
from scipy.sparse import csr_matrix
import operator
import logging
import io, pathlib, tempfile, tarfile, os


def directory_to_bytes(dirname):
    """Archive a whole directory to a stream of bytes using the tarfile package
      n.b. because this uses chdir, it is not generally thread safe
    """
    bio = io.BytesIO()
    path = os.path.abspath(os.getcwd())
    try:
        parent, directory = os.path.split(dirname)
        os.chdir(parent)
        with tarfile.open(fileobj=bio, mode='w:bz2') as tar:
            tar.add(directory, recursive=True)
        return bio.getvalue()
    finally:
        os.chdir(path)


def directory_from_bytes(dirname, tarball):
    """Restore a directory from bytes
          n.b. because this uses chdir, it is not generally thread safe
    """
    bio = io.BytesIO(tarball)
    path = os.path.abspath(os.getcwd())
    try:
        os.chdir(dirname)

        with tarfile.open(fileobj=bio, mode='r:bz2') as tar:
            tar.extractall(dirname)
            directories = os.listdir(dirname)
            assert len(directories) == 1
            return os.path.join(dirname, directories[0])
    finally:
        os.chdir(path)


class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self, active_dims=None):
        super().__init__(active_dims=active_dims)
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

         X is a matrix of fingerprints
          | mol_0_bit_0 mol_0_bit_1 ...  |
          | mol_1_bit_0 mol_1_bit_1 ...  |

        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X,
        # ends up being |bits_set_in_mol_0 bits_set_in_mol_1|
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        # ends up being |bits_set_in_mol_0 bits_set_in_mol_1|

        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2
        # ends up being a symmetric matrix where entry m_i_j is the number of bits in common
        #  between molecule_i and molecule_2

        # Analogue of denominator in Tanimoto formula
        # for more details of broadcasting_elementwise, see:  https://deeplizard.com/learn/video/QscEWm0QTRY
        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        kernel = self.variance * outer_product / denominator
        logging.debug("%s kernel shape: %r", self.__class__.__name__, kernel.shape)
        return kernel

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


class TanimotoGP:
    def __init__(self, kernel=None, maxiter=100):
        self.kernel = kernel or Tanimoto()
        self.maxiter = maxiter
        self.model = None  # The GPFlowModel
        self.y_scaler = StandardScaler()  # the scaler to use
        self.shape = None  # the shape of the training data

    def objective_closure(self):
        """Optimize the model to reduce the marginal likelihood
        This is the probability of generating the prediction (and sample) from the prior
        observed data data"""
        return -self.m.log_marginal_likelihood()

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        self.shape = (None, X_train.shape[1])
        y_train = np.array(y_train)
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1))
        self.m = gpflow.models.GPR(data=(X_train.astype(np.float64), y_train_scaled),
                                   mean_function=Constant(np.mean(y_train_scaled)),
                                   kernel=self.kernel,
                                   noise_variance=1)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.objective_closure,
                     self.m.trainable_variables,
                     options=dict(maxiter=self.maxiter))
        self.last_var = None

    def predict(self, X_test):
        """Returns predictions and variances"""
        X_test = np.array(X_test)
        if hasattr(self.m, 'predict_f_compiled'):
            predict_f = self.m.predict_f_compiled
        else:
            predict_f = self.m.predict_f
        mu, var = predict_f(X_test.astype(np.float64))
        y_pred = self.y_scaler.inverse_transform(mu)
        y_var = self.y_scaler.scale_ * var  # need to also rescale the variances!
        return y_pred.flatten(), y_var

    def __getstate__(self):
        """Allow pickling of the model"""
        assert self.shape
        d = {}
        d.update(self.__dict__)
        gpflow.utilities.freeze(self.m)
        self.m.predict_f_compiled = tf.function(
            self.m.predict_f, input_signature=[tf.TensorSpec(shape=self.shape, dtype=tf.float64)]
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            if self.m:
                tf.saved_model.save(self.m, tmpdirname)
            else:
                tf.saved_model.save(self.kernel, tmpdirname)

            bytes = directory_to_bytes(tmpdirname)

        if self.m:
            d['m'] = bytes
            d['kernel'] = None  # fit can't be called again?
        else:
            d['m'] = None
            d['kernel'] = bytes

        d['last_var'] = None
        return d

    def __setstate__(self, d):
        """Allow depickling of the model"""
        self.__dict__ = d
        with tempfile.TemporaryDirectory() as tmpdirname:
            if d['m']:
                dirname = directory_from_bytes(tmpdirname, d['m'])
                self.m = tf.saved_model.load(dirname)
                self.kernel = self.m.kernel
            else:
                dirname = directory_from_bytes(tmpdirname, d['kernel'])
                self.kernel = tf.saved_model.load(dirname)
                self.m = None


