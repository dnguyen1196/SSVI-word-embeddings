from Data.PMI import PMI_tensor
from Factorizer.SSVI import SSVI_Embedding_full
import sys
import pickle
import argparse



def do_synthetic_embeddings(num_words, order, output):
    PMI = PMI_tensor()
    PMI.synthesize_fake_PMI(num_words, order)

    factorizer = SSVI_Embedding_full(PMI)
    factorizer.produce_embeddings(output)

def do_embeddings_pmi(pickedfile, output):
    print ("Loading tensor...")
    with open(pickedfile, "rb") as f:
        pmi_tensor = pickle.load(f, encoding="latin1")
        num_words  = pmi_tensor.num_words
        order      = pmi_tensor.order
        print("Computing embeddings for ", num_words, " words from ", order, "th pmi tensor ... ")
        factorizer = SSVI_Embedding_full(pmi_tensor)
        factorizer.produce_embeddings(output)

parser = argparse.ArgumentParser(description='SSVI word embeddings')

group = parser.add_mutually_exclusive_group()
group.add_argument("-nwords", "--numwords", type=int, help="size of synthetic vocabulary")

parser.add_argument("-ord", "--order", type=int, help="order of PMI tensor")

group.add_argument("-p", "--pfile", type=str, help="Picked PMI object file")
parser.add_argument("-o", "--output", type=str, help="Output file")

args = parser.parse_args()

if args.numwords is not None:
    do_synthetic_embeddings(args.numwords, args.order, args.output)
if args.pfile is not None:
    do_embeddings_pmi(args.pfile, args.output)

