from Data.PMI import PMI_tensor
from Factorizer.SSVI_full import SSVI_Embedding_full
from Factorizer.SSVI_diag import SSVI_Embedding_Diag

num_words = 100
order    = 3

PMI = PMI_tensor()
PMI.synthesize_fake_PMI(num_words, 3)

# factorizer = SSVI_Embedding_full(PMI)
factorizer = SSVI_Embedding_Diag(PMI)

factorizer.produce_embeddings(None)
