from xml.etree import ElementTree as ET
import os
import glob
from tqdm import tqdm
import argparse

def clean_blog_text():
    """
    Cleans blog text

    """
    pass

def main():
    parser = argparse.ArgumentParser(description="text parser and cleaner")
    parser.add_argument("--data_path", type=str, required=True,
                        help="path where your files are")
    args = parser.parse_args()
    for file in tqdm(glob.glob(os.path.join(args.data_path, "**", "*.xml"), recursive=True)):
        tree = ET.parse(file)
        print(tree)
        break


if __name__=="__main__":
    main()
