import numpy as np
from abc import abstractclassmethod, abstractmethod


class PosteriorParam(object):
    def __init__(self, dims, D, initMean=None, initCov=None):
        self.dims = dims
        self.D    = D
        self.initMean = initMean
        self.initCov  = initCov
        self.params = [[] for _ in self.dims]

        if initMean is None or initCov is None:
            initMean, initCov = self.default_params(D)

        for i, s in enumerate(self.dims):
            self.params[i] = self.initialize_params(s, initMean, initCov)

    @abstractmethod
    def default_params(self, D):
        raise NotImplementedError

    @abstractmethod
    def initialize_params(self, nparams, mean, cov):
        raise NotImplementedError

    @abstractmethod
    def get_vector_distribution(self, dim, i):
        raise NotImplementedError

    @abstractmethod
    def update_vector_distribution(self, dim, i, m, S):
        raise NotImplementedError

    @abstractmethod
    def save_mean_params(self, dim, filename):
        raise NotImplementedError


class PosteriorFullCovariance(PosteriorParam):
    def __init__(self, dims, D, initMean=None, initCov=None):
        super(PosteriorFullCovariance, self).__init__(dims, D, initMean, initCov)

    def default_params(self, D):
        return np.ones((D,)), np.eye(D)

    def initialize_params(self, nparams, mean, cov):
        matrices = np.zeros((nparams, self.D + 1, self.D))

        for i in range(nparams):
            matrices[i, 0, :]   = np.copy(mean)
            matrices[i, 1 :, :] = np.copy(cov)

        return matrices

    def get_vector_distribution(self, dim, i):
        return self.params[dim][i, 0, :], self.params[dim][i, 1 : , :]

    def update_vector_distribution(self, dim, i, m, S):
        self.params[dim][i, 0, :] = m
        self.params[dim][i, 1:, :] = S

    def save_mean_params(self, dim, filename):
        np.savetxt(filename, self.params[dim][:, 0, :], delimiter=",")


class PosteriorDiagonalCovariance(PosteriorParam):
    def __init__(self, dims, D, initMean=None, initCov=None):
        super(PosteriorDiagonalCovariance, self).__init__(dims, D, initMean, initCov)

    def default_params(self, D):
        return np.ones((D,)), np.ones((D,))

    def initialize_params(self, nparams, mean, cov):
        assert(cov.ndim == 1)
        matrices = np.zeros((nparams, 2, self.D))
        for i in range(nparams):
            matrices[i, 0, :] = np.copy(mean)
            matrices[i, 1, :] = np.copy(cov)
        return matrices

    def get_vector_distribution(self, dim, i):
        return self.params[dim][i, 0, :], self.params[dim][i, 1, :]

    def update_vector_distribution(self, dim, i, m, S):
        self.params[dim][i, 0, :] = m
        self.params[dim][i, 1, :] = S

    def save_mean_params(self, dim, filename):
        np.savetxt(filename, self.params[dim][:, 0, :], delimiter=",")