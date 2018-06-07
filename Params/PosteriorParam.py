import numpy as np

class ApproximatePosteriorParams(object):
    def __init__(self, dims, D, initMean=None, initCov=None):
        self.dims = dims
        self.D    = D
        self.initMean = initMean
        self.initCov  = initCov

        ndim = len(dims)
        self.params = [[] for _ in self.dims]
        if initMean is None or initCov is None:
            initMean = np.ones((self.D,))
            initCov  = np.eye(self.D)

        for i, s in enumerate(self.dims):
            self.params[i] = self.initialize_params(s, initMean, initCov)

    def initialize_params(self, nparams, mean, cov):
        matrices = np.zeros((nparams, self.D, self.D + 1))

        for i in range(nparams):
            matrices[i, :, 0]  = np.copy(mean)
            matrices[i, :, 1:] = np.copy(cov)

        return matrices

    def get_vector_distribution(self, dim, i):
        return self.matrices[dim, i, :, 0], self.matrices[dim, i, :, 1:]

    def update_vector_distribution(self, dim, i, m, S):
        self.matrices[dim, i, :, :] = np.hstack((m, S))