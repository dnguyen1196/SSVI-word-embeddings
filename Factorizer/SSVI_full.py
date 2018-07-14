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
            # TODO: this should be S + np.multiply(m, m)

        di = y/s * d_acc - 1./s * np.inner(D_acc, mi)
        Di = -1./s * D_acc

        return di, Di

    def estimate_di_Di_batch(self, id, mi, Si, ys, entries):
        """

        :param id:
        :param mi:
        :param Si:
        :param ys:    (num_samples, )
        :param entries: (num_samples, tensor_order)
        :return:
        """
        num_samples    = np.size(ys, axis=0)
        other_word_ids = [entry[1:] for entry in entries]
        di_sum  = np.zeros((self.D,))
        Di_sum  = np.zeros((self.D, self.D))
        s = self.sigma

        for i in range(num_samples):
            # dij, Dij     = self.estimate_di_Di_entry(other_word_ids[i])
            # di_sum      += ys[i]/s * dij - 1./s * np.inner(Dij, mi)
            # Di_sum      += -1./s  * Dij
            dij, Dij       = self.estimate_di_Di_add(other_word_ids[i], ys[i], s, mi)
            di_sum        += dij
            Di_sum        += Dij

        return di_sum, Di_sum

    def estimate_di_Di_entry(self, word_ids):
        """

        :param word_ids:
        :return:
        """
        d_acc = np.ones((self.D,))
        D_acc = np.ones((self.D,self.D))

        for word_id in word_ids:
            m, S = self.variational_posterior.get_vector_distribution(self.ndim, word_id)

            d_acc = np.multiply(d_acc, m)
            D_acc = np.multiply(D_acc, S + np.outer(m, m))

        return d_acc, D_acc

    def estimate_di_Di_add(self, word_ids, y, s, mi):
        d_ijk = np.copy(mi)
        ms    = np.ones((self.D,))
        D_ijk = np.ones((self.D,self.D))

        for word_id in word_ids:
            m, S = self.variational_posterior.get_vector_distribution(self.ndim, word_id)

            # d_ijk.shape == (self.D,)
            # 1/s (S + mm^T)(S + mm^T) mi + (m . m) yij
            d_ijk = np.dot(S, d_ijk) + np.multiply(m, np.dot(m, d_ijk))
            ms    = np.multiply(ms, m)

            # Dijk.shape == (self.D,)
            D_ijk = np.multiply(D_ijk, S + np.outer(m, m))

        d_ijk = 1/s * (np.multiply(ms, y) - d_ijk)
        D_ijk = -1/s * D_ijk

        return d_ijk, D_ijk