import numpy as np
import time
import itertools

class PMI_matrix():
    def __init__(self):
        self.order = 2

    def read_from_csv_pmi(self, filename, repeat=False):
        """
        :param filename: filename containing the pmi_tensor pair
        It must be the case that the file arranges the word
        id in increasing order?

        The format of csv file must be
        header: num_words
        Body: <id1, id2, PMI>

        :return:
        """
        with open(filename, "r") as f:
            header = f.readline().rstrip()
            self.num_words = int(header)
            self.observations = [[[] for _ in range(self.num_words)] for _ in range(self.order)]
            
            #print(self.observations)
            for line in f:
                data = line.rstrip().split(",")
                #print(line)
                idx  = [int(x) for x in data[:-1]]
                pmi  = float(data[-1])

                if repeat:
                    set_perms = [idx]
                else:
                    set_perms = itertools.permutations(idx)

                for perm in set_perms:
                    for dim, col in enumerate(perm):
                        self.observations[dim][col].append((idx, pmi))

    def synthesize_PPMI_matrix(self, num_words, D=50, sparsity=0.1):
        self.num_words = num_words
        print("Generating synthetic pmi_tensor matrix ... ")
        start = time.time()

        m = np.ones((D,))
        S = np.eye(D)
        word_vectors = np.zeros((num_words, D))
        context_vectors = np.zeros((num_words, D))

        for i in range(num_words):
            word_vectors[i, :] = np.transpose(np.random.multivariate_normal(m, S))

        for i in range(num_words):
            context_vectors[i, :] = np.transpose(np.random.multivariate_normal(m, S))

        total = num_words ** 2 # Total number of entries in this tensor
        num_observed = int(total * sparsity)

        unique_coords = self.generate_unique_coords(num_observed, num_words)

        self.organize_observed_entries(unique_coords, word_vectors, context_vectors, num_words)

        end = time.time()
        print("Generating synthetic data took: ", end- start)

    def organize_observed_entries(self, unique_coords, word_vectors, context_vectors, num_words):
        _, D  = word_vectors.shape

        self.observations = [[[] for _ in range(num_words)] for _ in range(self.order)]

        # self.observations = np.zeros((self.order, num_words, D))
        # assert (np.size(self.observations, axis=0) == 2)

        for coord in unique_coords:
            w_id = coord[0]
            c_id = coord[1]

            pmi = np.sum(np.multiply(word_vectors[w_id, :], context_vectors[c_id, :]))

            # Record for both word and context vectors
            for dim, id in enumerate(coord):
                self.observations[dim][id].append((coord, pmi))

    def get_cooccurrence_list(self, dim, idx, subsampling=None, num_negatives=512):
        """
        :param idx:
        :return: The list of entries associated with the word vector[id]
        """
        if subsampling is None:
            positive_samples = self.observations[dim][idx]

            #return positive_samples
        else:
            sample_size = min(subsampling, len(self.observations[dim][idx]))
            if sample_size == 0:
                return []
            subsamples = np.random.choice(len(self.observations[dim][idx]), sample_size, replace=False)
            positive_samples = np.take(self.observations[dim][idx], subsamples, axis=0)

            #return positive_samples

        # TODO: generate negative samples and generalize to higher dimension
        negative_sample_idx = np.random.choice(self.num_words, num_negatives, replace=False)
        negative_samples    = []

        positive_idx     = {tuple(k[0]) for k in positive_samples}

        # TODO: make this more general for 3d counts
        for negative_idx in negative_sample_idx:
            coord = [negative_idx]
            #print(s.insert(dim, idx))
            coord.insert(dim, idx)
            if tuple(coord) in positive_idx:
                continue
    
            negative_samples.append((coord, 0.0))
        
        samples = np.concatenate((positive_samples, negative_samples), axis=0)
        return samples


    def generate_unique_coords(self, num_observed, num_word):
        total = np.power(num_word, self.order)
        idx = np.random.choice(total, num_observed, replace=False)

        unique_coords = []
        for id in idx:
            coord = self.id_to_coordinate(id, num_word)
            if len(set(coord)) == len(coord):
                unique_coords.append(coord)
        return unique_coords

    def id_to_coordinate(self, id, num_word):
        coord = []
        id = float(id)
        n  = self.order - 1
        for i in range(self.order):
            div = np.power(num_word, n)
            k = int(id / div)
            coord.append(k)
            id -= div * k
            n  -= 1
        return coord

