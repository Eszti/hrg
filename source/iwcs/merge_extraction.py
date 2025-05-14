import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--out", type=str)
    return parser


def merge(dir, out):
    extractions = []
    nr_extractions = 0

    for fn in sorted(os.listdir(dir)):
        if not fn.endswith(".allennlp"):
            continue
        with open(f"{dir}/{fn}", "r") as f:
            extractions.extend(f.read().split("\n"))
        nr_extractions += 1

    # Write grammar
    with open(f"{out}", "wt") as f:
        for extraction_line in extractions:
            if extraction_line:
                f.write(f"{extraction_line.strip()}\n")
    print(f"# extractions: {nr_extractions}")


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    merge(args.dir, args.out)
