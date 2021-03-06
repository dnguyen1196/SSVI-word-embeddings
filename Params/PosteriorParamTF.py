import tensorflow as tf
import numpy as np

class VariationalPosteriorParamsTF(object):
    def __init__(self, dims, D, initMean=None, initCov=None):
        self.dims = dims
        self.D    = D
        self.initMean = initMean
        self.initCov  = initCov

        self.params = [[] for _ in self.dims]
        self.tf_params = [[] for _ in self.dims]

        if initMean is None or initCov is None:
            initMean = np.ones((self.D,))
            initCov  = np.eye(self.D)

        for i, s in enumerate(self.dims):
            self.params[i] = self.initialize_params(s, initMean, initCov)
            self.tf_params[i] = tf.Variable(self.params[i])

    def initialize_params(self, nparams, mean, cov):
        matrices = np.zeros((nparams, self.D + 1, self.D))

        for i in range(nparams):
            matrices[i, 0, :]   = np.copy(mean)
            matrices[i, 1 :, :] = np.copy(cov)

        return matrices

    def get_vector_distribution(self, dim, i):
        return self.params[dim][i, 0, :], self.params[dim][i, 1 : , :]

    def update_vector_distribution(self, dim, i, m, S):
        self.params[dim][i, :, :] = np.vstack((m, S))
        self.tf_params[dim] = tf.assign(self.tf_params[dim], self.params[dim])

    def save_mean_params(self, dim, filename):
        # TODO: check the format for downstream embedding evaluation
        np.savetxt(filename, self.params[dim][:, 0, :])