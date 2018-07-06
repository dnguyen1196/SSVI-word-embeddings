import sys
import pickle
import argparse

parser = argparse.ArgumentParser(description='Align word embeddings')

group = parser.add_mutually_exclusive_group()

parser.add_argument("-p", "--pickle", type=str, help="Pickle file")
parser.add_argument("-e", "--embed", type=str, help="embedding file")

parser.add_argument("-o", "--output", type=str, help="Output embedding file")

args = parser.parse_args()

pickle_file = args.pickle
embedding_file = args.embed
output_file = args.output

embeddings = open(embedding_file, "r")
output     = open(output_file, "w")

with open(pickle_file, "rb") as f:
    aligner = pickle.load(f, encoding="latin1")
    word_id = 0
    for line in embeddings:
        word_embedding = line.rstrip()
        actual_word    = aligner[word_id]
        output.write(actual_word + " " + word_embedding)
        output.write("\n")
        word_id += 1


# python align_word_embeddings.py -e /home/mnguye16/SSVI/SSVI-word-embeddings/embeddings_results/embeddings_2016_126.txt
# -p /home/mnguye16/PMI_2016/wordPairPMI_2016wordIDs.pickle
# -o /aligned_embeddings/2016_embeddings_full.txt