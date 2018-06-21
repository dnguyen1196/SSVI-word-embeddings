from Params.PosteriorParamTF import VariationalPosteriorParamsTF
from Params.PosteriorParam import PosteriorFullCovariance
from Params.PosteriorParam import PosteriorDiagonalCovariance
from Factorizer.SSVI_interface import SSVI_interface
import numpy as np
from numpy.linalg import inv

class SSVI_Embedding_full(SSVI_interface):
    def __init__(self, pmi_tensor, D=50):
        super(SSVI_Embedding_full, self).__init__(pmi_tensor, D)

        self.variational_posterior = PosteriorFullCovariance([self.num_words], D)

    def init_di_Di(self):
        return np.ones((self.D,)), np.ones((self.D,self.D))

    def update_mean_param(self, word_id, m, S, di_acc):
        meanGrad = (np.multiply(self.pSigma_inv, self.pmu - m) + di_acc)
        meanStep = self.compute_stepsize_mean_param(word_id, meanGrad)
        m_next   = m + np.multiply(meanStep, meanGrad)
        return m_next

    def update_cov_param(self, word_id, m, S, Di_acc):
        covGrad = (np.diag(self.pSigma_inv) - 2 * Di_acc)
        covStep = 1/ (self.time_step + 1) # simple decreasing step size
        S_next = inv((np.ones_like(covGrad) - covStep) * inv(S) + np.multiply(covStep, covGrad))
        return S_next

    def estimate_di_Di(self, id, mi, Si, entry):
        """
        :param id:
        :param mi:
        :param Si:
        :param entry:
        :return:
        """
        coord = entry[0]
        y = entry[1]

        other_word_ids = []
        for i in coord[1:]:
            other_word_ids.append(i)

        d_acc = np.ones((self.D,))
        D_acc = np.ones((self.D,self.D))
        s = self.sigma

        for word_id in other_word_ids:
            m, S = self.variational_posterior.get_vector_distribution(self.ndim, word_id)

            d_acc = np.multiply(d_acc, m)
            D_acc = np.multiply(D_acc, S + np.outer(m, m))

        Di = -1./s * D_acc
        di = y/s * d_acc - 1./s * np.inner(D_acc, mi)
        return di, Di


class SSVI_embedding_diag(SSVI_interface):
    def __init__(self, pmi_tensor, D=50):
        super(SSVI_Embedding_full, self).__init__(pmi_tensor, D)

        self.variational_posterior = PosteriorDiagonalCovariance([self.num_words], D)

    def init_di_Di(self):
        return np.ones((self.D,)), np.ones((self.D,))

    def update_mean_param(self, word_id, m, S, di_acc):
        meanGrad = (np.multiply(self.pSigma_inv, self.pmu - m) + di_acc)
        meanStep = self.compute_stepsize_mean_param(word_id, meanGrad)
        m_next = m + np.multiply(meanStep, meanGrad)
        return m_next

    def update_cov_param(self, word_id, m, S, Di_acc):
        covGrad = (self.pSigma_inv - 2 * Di_acc)
        covStep = 1 / (self.time_step + 1)  # simple decreasing step size
        S_next = inv((np.ones_like(covGrad) - covStep) * inv(S) + np.multiply(covStep, covGrad))
        return S_next

    def estimate_di_Di(self, id, mi, Si, entry):
        """
        :param id:
        :param mi:
        :param Si:
        :param entry:
        :return:
        """
        coord = entry[0]
        y = entry[1]

        other_word_ids = []
        for i in coord[1:]:
            other_word_ids.append(i)

        d_acc = np.ones((self.D,))
        D_acc = np.ones((self.D, self.D))
        s = self.sigma

        for word_id in other_word_ids:
            m, S = self.variational_posterior.get_vector_distribution(self.ndim, word_id)
            d_acc = np.multiply(d_acc, m)
            D_acc = np.multiply(D_acc, S + np.outer(m, m))

        di = y * d_acc / s  - np.inner(D_acc, mi)/s
        Di = (np.square(d_acc) + Si ) / s
        return di, Di
