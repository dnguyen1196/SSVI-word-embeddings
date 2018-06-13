from Data.PMI import PMI_tensor
from Factorizer.SSVI import SSVI_Embedding
import sys
import pickle

if __name__ == "__main__":
    args = sys.argv[1:]
    picked_file = args[0]
    output     = args[1]

    with open(picked_file, "rb") as f:
        pmi_tensor = pickle.load(f, encoding="latin1")
        num_words  = pmi_tensor.num_words

        factorizer = SSVI_Embedding(num_words, pmi_tensor)
        factorizer.produce_embeddings(output)