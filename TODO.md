# 
- Close form update for the diagonal case
- Batch processing


# Test on word embeddings data


# Look into how to do evaluation
- Built in gensim tools that just needs a text filecontaining the words vectors
- 


# Implement GPU-powered code
- How to use GPU in tensor flow
- Parallelize the computation involving minibatches

# Other details
- Take note of the format of csvfile
- (0,0) might appear together so the code in SSVI might need to change



python align_word_embeddings.py -e /home/mnguye16/SSVI/SSVI-word-embeddings/embeddings_results/2016_pmi_pair_full/embeddings_2016_126.txt -p /home/mnguye16/PMI_2016/wordPairPMI_2016wordIDs.pickle -o /aligned_embeddings/2016_embeddings_full.txt