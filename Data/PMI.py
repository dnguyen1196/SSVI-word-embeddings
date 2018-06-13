import numpy as np
import time
import warnings


class PMI_tensor():
    def __init__(self):
        return

    def read_from_csv_pmi(self, filename):
        """
        :param filename: filename containing the pmi_tensor pair
        It must be the case that the file arranges the word
        id in increasing order?

        The format of csv file must be
        header: wordid, wordid, ..., pmi
        Body: <id1, id2, ..., PMI>

        :return:
        """
        self.observations = []
        id = -1
        self.num_words = 0

        with open(filename, "r") as f:
            header = f.readline().rstrip().split(",") # header
            self.order = len(header) - 1
            for line in f:
                data = line.rstrip().split(",")

                idx  = [int(x) for x in data[:-1]]
                pmi  = float(data[-1])

                if idx[0] > id: # Untracked word id
                    self.num_words += 1
                    self.observations.append([])
                    id += 1

                self.observations[id].append((idx, pmi))

    def synthesize_fake_PMI(self, num_words, order, D=50, sparsity=0.1):
        print("Generating synthetic pmi_tensor matrix ... ")
        start = time.time()

        m = np.ones((D,))
        S = np.eye(D)
        word_vectors = np.zeros((num_words, D))

        for i in range(num_words):
            word_vectors[i, :] = np.transpose(np.random.multivariate_normal(m, S))

        total = num_words ** order # Total number of entries in this tensor
        num_observed = int(total * sparsity)

        unique_coords = self.generate_unique_coords(num_observed, num_words, order)
        self.organize_observed_entries(unique_coords, word_vectors, num_words, order)

        end = time.time()
        print("Generating synthetic data took: ", end- start)

    def organize_observed_entries(self, unique_coords, word_vectors, num_words, order):
        _, D  = word_vectors.shape
        self.observations = [[] for _ in range(num_words)]

        for coord in unique_coords:
            m = np.ones((D,))
            for id  in coord:
                m = np.multiply(m, word_vectors[id, :])

            pmi = np.sum(m)

            for id in coord:
                self.observations[id].append((coord, pmi))

    def get_cooccurrence_list(self, id, subsampling=None):
        """
        :param id:
        :return: The list of entries associated with the word vector[id]
        """
        if subsampling is None:
            return self.observations[id]
        else:
            sample_size = min(subsampling, len(self.observations[id]))
            idx = np.random.choice(len(self.observations[id]), sample_size, replace=False)
            return np.take(self.observations[id], idx, axis=0)

    def generate_unique_coords(self, num_observed, num_word, order):
        total = np.power(num_word, order)
        idx = np.random.choice(total, num_observed, replace=False)

        unique_coords = []
        for id in idx:
            coord = self.id_to_coordinate(id, num_word, order)
            if len(set(coord)) == len(coord):
                unique_coords.append(coord)
        return unique_coords

    def id_to_coordinate(self, id, num_word, order):
        coord = []
        id = float(id)
        n  = order - 1
        for i in range(order):
            div = np.power(num_word, n)
            k = int(id / div)
            coord.append(k)
            id -= div * k
            n  -= 1
        return coord

