from Params.PosteriorParam import PosteriorDiagonalCovariance
from Factorizer.WC_factorizer_interface import WC_factorizer_interface
import numpy as np


class SSVI_WC_factorizer_diag(WC_factorizer_interface):
    def __init__(self, pmi_matrix, D=50):
        super(SSVI_WC_factorizer_diag, self).__init__(pmi_matrix, D)

        self.ada_cov = [np.zeros((self.num_words, D)) for _ in range(self.order)]
        self.variational_posterior = PosteriorDiagonalCovariance([self.num_words, self.num_words], D)
        self.cov_eta = 0.01

    def update_mean_param(self, dim, col, m, S, di_acc):
        meanGrad = (np.multiply(self.pSigma_inv, self.pmu - m) + di_acc)
        meanStep = self.compute_stepsize_mean_param(dim, col, meanGrad)
        m_next = m + np.multiply(meanStep, meanGrad)
        return m_next

    def update_cov_param(self, dim, col, m, S, Di_acc):
        covGrad = (self.pSigma_inv - 2 * Di_acc)
        covStep = self.compute_stepsize_cov_param(dim, col, covGrad)
        S_next = np.reciprocal((np.ones_like(covGrad) - covStep) * np.reciprocal(S) + np.multiply(covStep, covGrad))
        assert ((S_next > 0).all())
        return S_next

    def compute_stepsize_cov_param(self, dim, col, cGrad):
        acc_grad = self.ada_cov[dim][col, :]
        grad_sqr = np.square(cGrad)
        self.ada_cov[dim][col, :] = np.add(acc_grad, grad_sqr)
        return np.divide(self.cov_eta, np.sqrt(np.add(acc_grad, grad_sqr)))

    def estimate_di_Di_batch(self, dim, col, mi, Si, ys, coords):
        num_samples    = np.size(ys, axis=0)

        othercols_left     = coords[:, : dim]
        othercols_right    = coords[:, dim + 1 :]
        othercols          = np.concatenate((othercols_left, othercols_right), axis=1)

        alldims            = list(range(self.order))
        otherdims          = alldims[:dim]
        otherdims.extend(alldims[dim + 1 : ])

        di_sum  = np.zeros((self.D,))
        Di_sum  = np.zeros((self.D,))
        s = self.sigma

        for i in range(num_samples):
            dij, Dij       = self.estimate_di_Di_add(otherdims, othercols[i, :], ys[i], s, mi)
            di_sum        += dij
            Di_sum        += Dij

        return di_sum, Di_sum

    def estimate_di_Di_add(self, dims, cols, y, s, mi):
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

        for i, col in enumerate(cols):
            dim  = dims[i]
            m, S = self.variational_posterior.get_vector_distribution(dim, col)

            # d_ijk.shape == (self.D,)
            # 1/s (S + mm^T)(S + mm^T) mi + (m . m) yij
            d_ijk = np.multiply(S, d_ijk) + np.multiply(m, np.multiply(m, d_ijk))
            ms    = np.multiply(ms, m)

            # Dijk.shape == (self.D,)
            D_ijk = np.multiply(D_ijk, S + np.multiply(m, m))

        d_ijk = 1/s * (np.multiply(ms, y) - d_ijk)
        D_ijk = -1/s * D_ijk

        return d_ijk, D_ijk