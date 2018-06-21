from Data.PMI import PMI_tensor
from Factorizer.SSVI import SSVI_Embedding_full

num_words = 10
order    = 3

PMI = PMI_tensor()
PMI.synthesize_fake_PMI(num_words, 3)
filename = "random.txt"

factorizer = SSVI_Embedding_full(PMI)
factorizer.produce_embeddings(filename, report=100)
