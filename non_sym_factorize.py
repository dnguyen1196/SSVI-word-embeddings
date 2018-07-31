from Data.WC_matrix import PMI_matrix
from Factorizer.WC_factorizer_diag import SSVI_WC_factorizer_diag


import sys
import pickle
import argparse


def do_synthetic_embeddings(num_words, order, output, diag, D):
    if D is None:
        D = 50

    pmi_matrix = PMI_matrix()
    pmi_matrix.synthesize_PPMI_matrix(num_words, D)

    if diag:
        factorizer = SSVI_WC_factorizer_diag(pmi_matrix, D)
    else:
        factorizer = SSVI_WC_factorizer_diag(pmi_matrix, D)

    factorizer.produce_embeddings(output)

def do_embeddings_pmi(pickedfile, output, diag, D):
    print ("Loading tensor...")

    with open(pickedfile, "rb") as f:
        pmi_tensor = pickle.load(f, encoding="latin1")
        num_words  = pmi_tensor.num_words
        order      = pmi_tensor.order

        print("Computing embeddings for ", num_words, " words from ", order, "th pmi tensor ... ")
        if D is not None:
            D = 50

        if diag:
            factorizer = SSVI_WC_factorizer_diag(pmi_tensor, D)
        else:
            raise NotImplementedError

        factorizer.produce_embeddings(output)

parser = argparse.ArgumentParser(description='SSVI word embeddings')

group = parser.add_mutually_exclusive_group()
group.add_argument("-nwords", "--numwords", type=int, help="size of synthetic vocabulary")

parser.add_argument("-ord", "--order", type=int, help="order of PMI tensor")

group.add_argument("-p", "--pfile", type=str, help="Picked PMI object file")
parser.add_argument("-o", "--output", type=str, help="Output file")
parser.add_argument("--diag", action="store_true")
parser.add_argument("-d", "--dimension", type=int)

args = parser.parse_args()

if args.numwords is not None:
    do_synthetic_embeddings(args.numwords, args.order, args.output, args.diag, args.dimension)
if args.pfile is not None:
    do_embeddings_pmi(args.pfile, args.output, args.diag, args.dimension)

