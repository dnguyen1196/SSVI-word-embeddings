from Data.PMI import PMI_tensor
from Factorizer.SSVI import SSVI_Embedding

num_word = 10
order    = 3

PMI = PMI_tensor()
PMI.synthesize_fake_PMI(num_word, 3)

factorizer = SSVI_Embedding(num_word, PMI)
factorizer.factorize()
