from Data.WC_matrix import PMI_matrix
from Factorizer.WC_factorizer_diag import SSVI_WC_factorizer_diag

pmi_matrix = PMI_matrix()
num_words  = 100
sparsity   = 0.5

is_repeat = True

if not is_repeat:
    test_file = "test_pmi.txt"
else:
    test_file = "test_count_repeat.txt"

pmi_matrix.read_from_csv_pmi(test_file, repeat=is_repeat)
factorizer = SSVI_WC_factorizer_diag(pmi_matrix=pmi_matrix, D=80)

print(pmi_matrix.observations)

factorizer.produce_embeddings()
print(factorizer.variational_posterior.params[0])