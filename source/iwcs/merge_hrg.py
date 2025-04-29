import argparse
import os
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", type=str)
    parser.add_argument("--out", type=str)
    return parser


def merge(dirs, out):
    rules = []
    nr_rules = 0

    for d in dirs:
        for fn in os.listdir(d):
            if not fn.endswith(".hrg"):
                continue
            with open(f"{d}/{fn}", "r") as f:
                rules.extend(f.read().split("\n"))
            nr_rules += 1

    # Write grammar
    c = Counter(rules)
    weighted_rules = [
        (rule, float(c[rule]) / len(rules)) for rule, cnt in c.most_common()
    ]
    with open(f"grammar/{out}", "w") as f:
        for rule, w in weighted_rules:
            f.write(f"{rule};\t{w}\n")
    print(f"# rules: {nr_rules}")


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    merge(args.dirs, args.out)
