import numpy as np
import time

class PMI_tensor():
    def __init__(self):

        return


    def synthesize_fake_PMI(self, num_words, order, D=50, sparsity=0.1):
        print("Generating synthetic PMI matrix ... ")
        start = time.time()

        m = np.ones((D,))
        S = np.eye(D)
        word_vectors = np.zeros((num_words, D))

        for i in range(num_words):
            word_vectors[i, :] = np.random.multivariate_normal(m, S)

        total = num_words ** order # Total number of entries in this tensor
        num_observed = int(total * sparsity)

        unique_coords = self.generate_unique_coords(num_observed, num_words, order)
        self.organize_observed_entries(unique_coords, word_vectors, num_words)

        end = time.time()
        print("Generating synthetic data took: ", end- start)

    def organize_observed_entries(self, unique_coords, word_vectors, num_words):
        order = len(unique_coords[0])
        D     = len(word_vectors[0])
        self.observations = [[] for _ in range(num_words)]
        for coord in unique_coords:
            m = np.ones(D)
            for i  in range(order):
                m = np.multiply(m, word_vectors[i])

            pmi = np.sum(m)
            for i in range(order):
                self.observations[i].append((coord, pmi))

    def get_cooccurrence_list(self, id, subsampling=None):
        """
        :param id:
        :return: The list of entries associated with the word vector[id]
        """
        if subsampling is None:
            return self.observations[id]
        else:
            idx = np.random.choice(len(self.observations[id]), sampling_size, replace=False)
            return np.take(self.observations[id], idx, axis=0)

    def generate_unique_coords(self, num_observed, num_word, order):
        total = num_word ** order

        idx = np.random.choice(total, num_observed, replace=False)

        unique_coords = []
        for id in idx:
            unique_coords.append(self.id_to_coordinate(id, num_word, order))
        return unique_coords

    def id_to_coordinate(self, id, num_word, order):
        coord = []
        id = float(id)
        for _ in range(order):
            k = int(id / num_word)
            coord.append(k)
            id -= num_word * k
        return coord

