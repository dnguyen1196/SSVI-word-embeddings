from Data.WC_matrix import PMI_matrix
from Factorizer.WC_factorizer_diag import SSVI_WC_factorizer_diag

pmi_matrix = PMI_matrix()
num_words  = 100
sparsity   = 0.5

pmi_matrix.read_from_csv_pmi("test_pmi.txt")
factorizer = SSVI_WC_factorizer_diag(pmi_matrix=pmi_matrix, D=80)

factorizer.produce_embeddings()
