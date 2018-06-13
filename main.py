from Data.PMI import PMI_tensor
from Factorizer.SSVI import SSVI_Embedding
import pickle

num_word = 10
order    = 3

with open("./pmi_cleaned.p", "rb") as f:
    PMI = pickle.load(f, encoding="latin1")
    print(PMI.get_cooccurrence_list(0))
