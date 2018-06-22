from Params.PosteriorParam import PosteriorFullCovariance
from abc import abstractclassmethod, abstractmethod
import time
import numpy as np
from numpy.linalg import inv

class SSVI_interface(object):
    def __init__(self, pmi_tensor, D=50):
        self.num_words   = pmi_tensor.num_words
        self.D           = D

        self.pmi_tensor = pmi_tensor

        self.sigma = 1
        self.batch_size        = 1000
        self.ndim              = 0
        self.pSigma_inv        = np.ones((self.D,))
        self.pmu               = np.ones((self.D,))

        # Optimization variables
        self.eta               = 1
        self.ada_grad          = np.zeros((self.num_words, D))
        self.max_iterations    = 6001

        self.norm_changes      = np.zeros((self.num_words, 2))
        self.epsilon           = 0.001

    def produce_embeddings(self, filename=None, report=1, num_epochs = 500):
        self.report = report
        start = time.time()

        for epoch in range(num_epochs):
            for word_id in range(self.num_words):
                observed_i = self.pmi_tensor.get_cooccurrence_list(word_id, self.batch_size)
                m, S = self.variational_posterior.get_vector_distribution(self.ndim, word_id)

                di_acc, Di_acc = self.init_di_Di()

                for entry in observed_i:
                    di, Di = self.estimate_di_Di(word_id, m, S, entry)
                    di_acc += di
                    Di_acc += Di

                Di_acc *= len(observed_i) / min(self.batch_size, len(observed_i))
                di_acc *= len(observed_i) / min(self.batch_size, len(observed_i))

                S_next = self.update_cov_param(word_id, m, S, Di_acc)
                m_next = self.update_mean_param(word_id, m, S, di_acc)

                self.keep_track_changes(word_id, m, S, m_next, S_next)
                self.variational_posterior.update_vector_distribution(self.ndim, word_id, m_next, S_next)

            self.time_step += 1
            delta_m, delta_c = self.check_stopping_condition()
            self.report_convergence(epoch, delta_m, delta_c, start)
            if max(delta_m, delta_c) < self.epsilon:  # Stopping criterion
                break
            # After each epoch, save the embeddings
            if filename is not None:
                self.save_embeddings(filename + "_" + str(epoch) + ".txt")

    @abstractmethod
    def init_di_Di(self):
        raise NotImplementedError

    @abstractmethod
    def update_mean_param(self, word_id, m, S, di_acc):
        raise NotImplementedError

    @abstractmethod
    def update_cov_param(self, word_id, m, S, Di_acc):
        raise NotImplementedError

    @abstractmethod
    def estimate_di_Di(self, id, mi, Si, entry):
        raise NotImplementedError

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

    def keep_track_changes(self, id, m, S, m_next, S_next):
        delta_m = np.linalg.norm(m_next - m)
        if S.ndim == 1:
            delta_c = np.linalg.norm(S_next - S)
        else:
            delta_c = np.linalg.norm(S_next - S, "fro")
        self.norm_changes[id, :] = np.array([delta_m, delta_c])

    def check_stopping_condition(self):
        """
        :return:
        """
        max_delta_m = 0.
        max_delta_c = 0.
        for id in range(self.num_words):
            delta_m, delta_c = self.norm_changes[id, :]
            max_delta_m = max(delta_m, max_delta_m)
            max_delta_c = max(delta_c, max_delta_c)

        return max_delta_m, max_delta_c

    def report_convergence(self, epoch, delta_m, delta_c, start):
        if epoch == 0:
            print("   epoch  | delta_m  | delta_c  | time ")
        print('{:^10} {:^10} {:^10}'.format(epoch, np.around(delta_m, 4), \
                                            np.around(delta_c, 4)),\
                                            np.around(time.time() - start, 2))

    def save_embeddings(self, filename):
        self.variational_posterior.save_mean_params(self.ndim, filename)
