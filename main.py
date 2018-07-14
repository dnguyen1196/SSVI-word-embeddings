from Data.PMI import PMI_tensor
from Factorizer.SSVI_full import SSVI_Embedding_full
from Factorizer.SSVI_diag import SSVI_Embedding_Diag
import sys
import pickle
import argparse


def do_synthetic_embeddings(num_words, order, output, diag, D):
    pmi_tensor = PMI_tensor()
    pmi_tensor.synthesize_fake_PMI(num_words, order)
    if D is None:
        D = 50

    if diag:
        factorizer = SSVI_Embedding_Diag(pmi_tensor, D)
    else:
        factorizer = SSVI_Embedding_full(pmi_tensor, D)
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
            factorizer = SSVI_Embedding_Diag(pmi_tensor, D)
        else:
            factorizer = SSVI_Embedding_full(pmi_tensor, D)
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

