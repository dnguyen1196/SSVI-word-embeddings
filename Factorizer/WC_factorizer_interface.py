from abc import abstractmethod
import time
import numpy as np


class WC_factorizer_interface(object):
    def __init__(self, pmi_tensor, D=50):
        self.num_words   = pmi_tensor.num_words
        self.D           = D
        self.order       = 2
        self.pmi_tensor = pmi_tensor

        self.sigma = 1
        self.batch_size        = 1024
        self.negative_num      = 1024
        self.ndim              = 2 # change this to 2
        self.pSigma_inv        = np.ones((self.D,))
        self.pmu               = np.ones((self.D,))

        # Optimization variables
        self.grad_eta          = 0.01
        self.ada_grad          = [np.zeros((self.num_words, D)) for _ in range(self.order)]
        self.max_iterations    = 6001
        self.time_step         = 1

        self.word_dimension    = 0
        self.norm_changes      = [np.zeros((self.num_words, 3)) for _ in range(self.order)]
        self.epsilon           = 0.0001

    def produce_embeddings(self, filename=None, num_epochs = 500):
        start = time.time()

        for epoch in range(num_epochs):
            for col in range(self.num_words):
                for dim in range(self.ndim):
                    # include both negative and positive examples
                    observations = self.pmi_tensor.get_cooccurrence_list(dim, col, self.batch_size, self.negative_num)

                    if len(observations) == 0:
                        continue

                    # get vector distribution for the word/context vector
                    m, S         = self.variational_posterior.get_vector_distribution(dim, col)

                    # Get the list of ys and coords (batch)
                    ys = np.array([entry[1] for entry in observations])
                    coords = np.array([entry[0] for entry in observations])

                    di_acc, Di_acc = self.estimate_di_Di_batch(dim, col, m, S, ys, coords)

                    Di_acc *= len(observations) / min(self.batch_size, len(observations))
                    di_acc *= len(observations) / min(self.batch_size, len(observations))

                    S_next = self.update_cov_param(dim, col, m, S, Di_acc)
                    m_next = self.update_mean_param(dim, col, m, S, di_acc)

                    self.keep_track_changes(dim, col, m, S, m_next, S_next)
                    self.variational_posterior.update_vector_distribution(dim, col, m_next, S_next)

            self.time_step += 1
            delta_m_nat, delta_m, delta_c = self.check_stopping_condition()

            self.report_convergence(epoch, delta_m_nat, delta_m, delta_c, start)
            # After each epoch, save the embeddings
            if filename is not None:
                self.save_embeddings(filename + "_" + str(epoch) + ".txt")

            if self.satisfy_stopping_condition(delta_m_nat, delta_m, delta_c):
                # Stopping criterion
                break

    @abstractmethod
    def update_mean_param(self, dim, col, m, S, di_acc):
        raise NotImplementedError

    @abstractmethod
    def update_cov_param(self, dim, col, m, S, Di_acc):
        raise NotImplementedError

    @abstractmethod
    def estimate_di_Di_batch(self, dim, col, mi, Si, ys, entries):
        raise NotImplementedError

    def compute_stepsize_mean_param(self, dim, col, mGrad):
        """
        :param col: dimension of the hidden matrix
        :param i: column of hidden matrix
        :param mGrad: computed gradient
        :return:

        Compute the update for the mean parameter dependnig on the
        optimization scheme
        """
        acc_grad = self.ada_grad[dim][col, :]
        grad_sqr = np.square(mGrad)
        self.ada_grad[dim][col, :] += grad_sqr
        return np.divide(self.grad_eta, np.sqrt(np.add(acc_grad, grad_sqr)))

    def keep_track_changes(self, dim, col, m, S, m_next, S_next):
        # Changes in the standard space
        delta_m     = np.linalg.norm(m_next - m)
        if S.ndim == 1:
            delta_c = np.linalg.norm(S_next - S)
            delta_m_nat = np.linalg.norm(np.multiply(np.reciprocal(S_next), m_next) \
                                         - np.multiply(np.reciprocal(S), m))
        else:
            delta_c = np.linalg.norm(S_next - S, "fro")
            delta_m_nat = np.linalg.norm(np.inner(S_next, m_next) - np.inner(S, m))

        self.norm_changes[dim][col, :] = np.array([delta_m_nat, delta_m, delta_c])

    def check_stopping_condition(self):
        """
        :return:
        """
        max_delta_m = 0.
        max_delta_c = 0.
        max_delta_nat = 0.
        for col in range(self.num_words):
            # The first dimension
            delta_nat, delta_m, delta_c = self.norm_changes[self.word_dimension][col, :]
            max_delta_m = max(delta_m, max_delta_m)
            max_delta_c = max(delta_c, max_delta_c)
            max_delta_nat = max(delta_nat, max_delta_nat)

        return max_delta_nat, max_delta_m, max_delta_c

    def report_convergence(self, epoch, delta_m_nat, delta_m, delta_c, start):
        if epoch == 0:
            print("   epoch  | delta_nat| delta_m  | delta_c  | time ")
        print('{:^10} {:^10} {:^10} {:^10}'.format(epoch, np.around(delta_m, 4), \
                                            np.around(delta_m_nat, 4),\
                                            np.around(delta_c, 4)),\
                                            np.around(time.time() - start, 2))

    def save_embeddings(self, filename):
        self.variational_posterior.save_mean_params(self.word_dimension, filename)

    def satisfy_stopping_condition(self, delta_m_nat, delta_m, delta_c):
        return max(delta_m_nat, delta_m) < self.epsilon
