import sys
import pickle
import argparse

parser = argparse.ArgumentParser(description='Align word embeddings')

group = parser.add_mutually_exclusive_group()

parser.add_argument("-p", "--pickle", type=str, help="Pickle file")
parser.add_argument("-e", "--embed", type=str, help="embedding file")

parser.add_argument("-o", "--output", type=str, help="Output embedding file")

args = parser.parse_args()

pickle_file = parser.pickle
embedding_file = parser.embed
output_file = parser.output

embeddings = open(embedding_file, "r")
output     = open(output_file, "w")

with open(pickle_file, "rb") as f:
    aligner = pickle.load(f, encoding="latin1")
    word_id = 0
    for line in embeddings:
        word_embedding = line.rstrip()
        actual_word    = aligner[word_id]
        output.write(actual_word + " " + word_embedding)
        word_id += 1


