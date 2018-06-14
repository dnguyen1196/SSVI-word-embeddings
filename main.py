from Data.PMI import PMI_tensor
from Factorizer.SSVI import SSVI_Embedding
import sys
import pickle
import argparse



def do_synthetic_embeddings(num_words, order, output):
    PMI = PMI_tensor()
    PMI.synthesize_fake_PMI(num_words, order)

    factorizer = SSVI_Embedding(PMI)
    factorizer.produce_embeddings(output, report=100)

def do_embeddings_pmi(pickedfile, output):
    print ("Loading tensor...")
    with open(picked_file, "rb") as f:
        pmi_tensor = pickle.load(f, encoding="latin1")
        num_words  = pmi_tensor.num_words
        print("Factorizing ... ")
        factorizer = SSVI_Embedding(pmi_tensor)
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

