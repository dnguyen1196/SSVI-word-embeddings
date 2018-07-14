from Params.PosteriorParam import PosteriorDiagonalCovariance
from Factorizer.SSVI_interface import SSVI_interface
import numpy as np


class SSVI_Embedding_Diag(SSVI_interface):
    def __init__(self, pmi_tensor, D=50):
        super(SSVI_Embedding_Diag, self).__init__(pmi_tensor, D)

        self.ada_cov = np.zeros((self.num_words, D))
        self.variational_posterior = PosteriorDiagonalCovariance([self.num_words], D)
        self.cov_eta = 0.01

    def init_di_Di(self):
        return np.ones((self.D,)), np.ones((self.D,))

    def update_mean_param(self, word_id, m, S, di_acc):
        meanGrad = (np.multiply(self.pSigma_inv, self.pmu - m) + di_acc)
        meanStep = self.compute_stepsize_mean_param(word_id, meanGrad)
        m_next = m + np.multiply(meanStep, meanGrad)
        return m_next

    def update_cov_param(self, word_id, m, S, Di_acc):
        covGrad = (self.pSigma_inv - 2 * Di_acc)
        covStep = self.compute_stepsize_cov_param(word_id, covGrad)
        S_next = np.reciprocal((np.ones_like(covGrad) - covStep) * np.reciprocal(S) + np.multiply(covStep, covGrad))
        assert ((S_next > 0).all())
        return S_next

    def compute_stepsize_cov_param(self, id, cGrad):
        acc_grad = self.ada_cov[id, :]
        grad_sqr = np.square(cGrad)
        self.ada_cov[id, :] = np.add(acc_grad, grad_sqr)
        return np.divide(self.cov_eta, np.sqrt(np.add(acc_grad, grad_sqr)))

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
        s     = self.sigma

        for word_id in other_word_ids:
            m, S = self.variational_posterior.get_vector_distribution(self.ndim, word_id)
            d_acc = np.multiply(d_acc, m)
            # D_acc = np.multiply(D_acc, S + np.outer(m, m))
            D_acc = np.multiply(D_acc,  S + np.multiply(m, m))
            # print(D_acc.shape)

        di = y * d_acc / s  - np.inner(D_acc, mi)/s
        Di = (np.square(d_acc) + Si ) / s

        # print(Di.shape)
        return di, Di

    def estimate_di_Di_batch(self, id, mi, Si, ys, entries):
        num_samples    = np.size(ys, axis=0)
        other_word_ids = [entry[1:] for entry in entries]

        di_sum  = np.zeros((self.D,))
        Di_sum  = np.zeros((self.D,))
        s = self.sigma

        for i in range(num_samples):
            # dij, Dij     = self.estimate_di_Di_entry(other_word_ids[i])
            # dij.shape == Dij.shape == (self.D,)
            # di_sum      += ys[i] * dij / s  - np.multiply(Dij, mi)/s
            # Di_sum      += Dij / -s
            dij, Dij       = self.estimate_di_Di_add(other_word_ids[i], ys[i], s, mi)
            di_sum        += dij
            Di_sum        += Dij

        return di_sum, Di_sum

    def estimate_di_Di_entry(self, word_ids):
        """
        :param word_ids:
        :return:
        """
        d_ijk = np.ones((self.D,))
        S_mm  = np.ones((self.D,))
        D_ijk = np.ones((self.D,))

        for word_id in word_ids:
            m, S = self.variational_posterior.get_vector_distribution(self.ndim, word_id)

            d_ijk = np.multiply(d_ijk, m)
            D_ijk = np.multiply(D_ijk, S + np.multiply(m, m))

        return d_ijk, D_ijk

    def estimate_di_Di_add(self, word_ids, y, s, mi):
        """

        :param word_id:
        :param y:
        :param s:
        :param mi:
        :return:
        """
        d_ijk = np.copy(mi)
        ms    = np.ones((self.D,))
        D_ijk = np.ones((self.D,))

        for word_id in word_ids:
            m, S = self.variational_posterior.get_vector_distribution(self.ndim, word_id)

            # d_ijk.shape == (self.D,)
            # 1/s (S + mm^T)(S + mm^T) mi + (m . m) yij
            d_ijk = np.multiply(S, d_ijk) + np.multiply(m, np.multiply(m, d_ijk))
            ms    = np.multiply(ms, m)

            # Dijk.shape == (self.D,)
            D_ijk = np.multiply(D_ijk, S + np.multiply(m, m))

        d_ijk = 1/s * (np.multiply(ms, y) - d_ijk)
        D_ijk = -1/s * D_ijk

        return d_ijk, D_ijk