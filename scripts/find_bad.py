import numpy as np
import argparse
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser("Simple script to filter results")
parser.add_argument("--input", type=str, default="../build/statOut/ImageFreq.csv")
parser.add_argument("--classes", type=str, required=True)
parser.add_argument("--thresh", type=int, default=10000)

def get_bad(filename, classes, thresh):
    paths = set()
    #import pdb; pdb.set_trace()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in (reader):
            for class_id in classes:
                if int(row[str(class_id)]) > thresh:
                    paths.add(os.path.dirname(row["path"]))
                    break
    return paths

def main():
    args = parser.parse_args()
    classes = list(map(int, args.classes.split(",")))

    bad_set = get_bad(args.input, classes, args.thresh)

    for v in bad_set:
        print(v)

    # if args.count:
    #     print(len(files), "({0:.2f}%)".format(100*len(files)/lines))
    # else:
    #     print("\n".join(files))


if __name__ == '__main__':
    main()
