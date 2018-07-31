# Why is the right way to do decomposition
- What sort of PMI matrix/PPMI?

# command to learn word embeddings

python main.py
-p <PMI pickled file>
-o <output file path, note that it will output file with this name + epoch count>
-d <dimension of word vector>
--diag (optional diagonal covariance vs full covariance)

Diagonal covariance
python main.py -p pmi_2016_full.p -o ./embeddings_results/2016_pmi_pair_diag/2016_pair_pmi --diag -d 80

Full covariance
python main.py -p pmi_2016_full.p -o ./embeddings_results/2016_pmi_pair_diag/2016_pair_pmi --diag -d 80

# Command to align embeddings
python align_word_embeddings.py -e ./embeddings_results/2016_pmi_pair_diag/2016_pair_pmi_181.txt -p /home/mnguye16/PMI_2016/wordPairPMI_2016wordIDs.pickle  -o ./aligned_embeddings/2016_embeddings_diag.txt


/home/mnguye16/PMI_2016