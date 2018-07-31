# Implementation problem

- still no negative pmi
- test on the recently obtained vectors
- try word context pair factorization instead of word x word x word factorization
-
- 

# TODO 
- Batch processing
- No way to learn all words, increase the min word count

# Test on word embeddings data
- What to do in the case of word id with no available observed data?
- Just keep a list of observed word index to look at
And then in the internal representation, keep a mapping from
word_id -> actual place in memory
-

# Look into how to do evaluation
- May be have a seperate project to do evaluation given embedding file 


# Implement GPU-powered code
- How to use GPU in tensor flow
- Parallelize the computation involving minibatches

# Other details

python align_word_embeddings.py -e /home/mnguye16/SSVI/SSVI-word-embeddings/embeddings_results/2016_pmi_pair_full/embeddings_2016_126.txt -p /home/mnguye16/PMI_2016/wordPairPMI_2016wordIDs.pickle -o /aligned_embeddings/2016_embeddings_full.txt