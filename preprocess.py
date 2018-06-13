"""
From csv file preprocess the data
And convert the PMI data into picked-tensor PMI object
"""

import pickle
import sys
from Data.PMI import PMI_tensor

print("o")

if __name__ == "__main__":
    print("eh")
    names = sys.argv[1:]
    input_file = names[0]
    output_file = names[1]
    pmi_tensor = PMI_tensor()
    pmi_tensor.read_from_csv_pmi(input_file)
    pickle.dump(pmi_tensor, open(output_file, "wb"))