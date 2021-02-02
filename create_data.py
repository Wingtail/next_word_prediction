from nltk.corpus import gutenberg
from tqdm import tqdm
import pickle
import csv
import random
import os
import numpy as np
from nwp_dataset_creator import NWPDataCreator
import glob
def main():
    data_creator = NWPDataCreator()
    data_dirs = glob.glob(os.path.join("./text_data/","*.txt"))

    data_creator.create_data(data_dirs)



if __name__ == '__main__':
    main()

