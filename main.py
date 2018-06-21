from Data.PMI import PMI_tensor
from Factorizer.SSVI import SSVI_Embedding_full
from Factorizer.SSVI import SSVI_Embedding_Diag
import sys
import pickle
import argparse


def do_synthetic_embeddings(num_words, order, output, diag):
    PMI = PMI_tensor()
    PMI.synthesize_fake_PMI(num_words, order)
    if diag:
        factorizer = SSVI_Embedding_Diag(PMI)
    else:
        factorizer = SSVI_Embedding_full(PMI)
    factorizer.produce_embeddings(output)

def do_embeddings_pmi(pickedfile, output, diag):
    print ("Loading tensor...")
    with open(pickedfile, "rb") as f:
        pmi_tensor = pickle.load(f, encoding="latin1")
        num_words  = pmi_tensor.num_words
        order      = pmi_tensor.order
        print("Computing embeddings for ", num_words, " words from ", order, "th pmi tensor ... ")
        if diag:
            factorizer = SSVI_Embedding_Diag(PMI)
        else:
            factorizer = SSVI_Embedding_full(PMI)
        factorizer.produce_embeddings(output)

parser = argparse.ArgumentParser(description='SSVI word embeddings')

group = parser.add_mutually_exclusive_group()
group.add_argument("-nwords", "--numwords", type=int, help="size of synthetic vocabulary")

parser.add_argument("-ord", "--order", type=int, help="order of PMI tensor")

group.add_argument("-p", "--pfile", type=str, help="Picked PMI object file")
parser.add_argument("-o", "--output", type=str, help="Output file")
parser.add_argument("--diag", action="store_true")
args = parser.parse_args()

if args.numwords is not None:
    do_synthetic_embeddings(args.numwords, args.order, args.output, args.diag)
if args.pfile is not None:
    do_embeddings_pmi(args.pfile, args.output, args.diag)

