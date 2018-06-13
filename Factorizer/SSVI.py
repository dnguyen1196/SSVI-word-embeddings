from Params.PosteriorParamTF import VariationalPosteriorParamsTF
import numpy as np
from numpy.linalg import inv

class SSVI_Embedding(object):
    def __init__(self, pmi_tensor, D=50):
        self.num_word   = pmi_tensor.num_words
        self.D          = D
        self.variational_posterior = VariationalPosteriorParamsTF([num_word], D)

        self.pmi_tensor = pmi_tensor

        self.sigma = 1
        self.batch_size        = 128
        self.ndim              = 0
        self.pSigma_inv        = np.eye(self.D,)
        self.pmu               = np.ones((self.D,))

        # Optimization variables
        self.eta = 1
        self.ada_grad = np.zeros((num_word, D))
        self.max_iterations = 6001
        self.time_step = np.ones((num_word,))

    def factorize(self):
        for iter in range(self.max_iterations):
            word_id = iter % self.num_word
            observed_i = self.pmi_tensor.get_cooccurrence_list(word_id, self.batch_size)
            m, S = self.variational_posterior.get_vector_distribution(self.ndim, word_id)

            di_acc = np.zeros((self.D,))
            Di_acc = np.zeros((self.D, self.D))

            for entry in observed_i:
                di, Di = self.estimate_di_Di(word_id, m, S, entry)
                di_acc += di
                Di_acc += Di

            Di_acc *= len(observed_i) / min(self.batch_size, len(observed_i))
            di_acc *= len(observed_i) / min(self.batch_size, len(observed_i))

            S_next = self.update_cov_param(word_id, m, S, Di_acc)
            m_next = self.update_mean_param(word_id, m, S, di_acc)

            self.variational_posterior.update_vector_distribution(self.ndim, word_id, m_next, S_next)
            self.time_step[word_id] += 1

    def update_mean_param(self, word_id, m, S, di_acc):
        meanGrad = (np.inner(self.pSigma_inv, self.pmu - m) + di_acc)
        meanStep = self.compute_stepsize_mean_param(word_id, meanGrad)
        m_next   = m + np.multiply(meanStep, meanGrad)
        return m_next

    def update_cov_param(self, word_id, m, S, Di_acc):
        covGrad = (self.pSigma_inv - 2 * Di_acc)
        covStep = 1/ (self.time_step[word_id] + 1) # simple decreasing step size
        S_next = inv((np.ones_like(covGrad) - covStep) * inv(S) + np.multiply(covStep, covGrad))
        return S_next

    def compute_stepsize_mean_param(self, id, mGrad):
        """
        :param id: dimension of the hidden matrix
        :param i: column of hidden matrix
        :param mGrad: computed gradient
        :return:

        Compute the update for the mean parameter dependnig on the
        optimization scheme
        """
        acc_grad = self.ada_grad[id, :]
        grad_sqr = np.square(mGrad)
        self.ada_grad[id, :] += grad_sqr
        return np.divide(self.eta, np.sqrt(np.add(acc_grad, grad_sqr)))

    def stopping_condition(self):
        """
        :return:
        """
        return False

    def save_embeddings(self, filename):
        self.variational_posterior.save_mean_params(self.ndim, filename)

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
        for i in coord:
            if i != id:
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