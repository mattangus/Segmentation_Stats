import numpy as np
import argparse
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Simple script to filter results")
parser.add_argument("--input", type=str, default="../build/statOut/ImageFreq.csv")
parser.add_argument("--class_id", type=int, required=True)
parser.add_argument("--thresh", type=int, default=0)
parser.add_argument("--count", action="store_true", default=False)
parser.add_argument("--plot", action="store_true", default=False)

BAD_PATHS = ["p5_clear/T17_13-15", "p5_clear/T19_14-13",
            "p9_clear/T27_19-10", "p5_clear/T20_14-31"]

def get_list(filename, str_class, thresh):
    files = []
    lines = 0
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in (reader):
            if int(row[str_class]) > thresh and "p5" not in row["path"]:
                has_bad = any([p in row["path"] for p in BAD_PATHS])
                if not has_bad:
                    files.append(row["path"]) 
            lines += 1
    
    return files, lines

def plot(filename, str_class):

    threshs = np.arange(0, 5000, 10)
    counts = np.zeros(len(threshs))
    lines = 0
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in (reader):
            counts += int(row[str_class]) <= threshs
            lines += 1

    pct = counts/lines
    plt.plot(threshs, pct)
    plt.show()
    plt.plot(threshs, np.gradient(pct))
    plt.show()

def main():
    args = parser.parse_args()
    str_class = str(args.class_id)

    if args.plot:
        plot(args.input, str_class)
    else:
        files, lines = get_list(args.input, str_class, args.thresh)

        if args.count:
            print(len(files), "({0:.2f}%)".format(100*len(files)/lines))
        else:
            print("\n".join(files))


if __name__ == '__main__':
    main()
