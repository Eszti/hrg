import json
import os
from collections import Counter


def merge():
    rules = []
    overlapping_subgraph = []
    nr_rules = 0

    for fn in os.listdir("train_out"):
        if not fn.endswith(".hrg"):
            continue
        with open(f"train_out/{fn}", "r") as f:
            rules.extend(f.read().split("\n"))
        nr_rules += 1

    # Write grammar
    c = Counter(rules)
    weighted_rules = [
        (rule, float(c[rule]) / len(rules)) for rule, cnt in c.most_common()
    ]
    with open("grammar/lsoie_wiki.hrg", "w") as f:
        for rule, w in weighted_rules:
            f.write(f"{rule};\t{w}\n")
    with open("train_out/overlapping.txt", "w") as f:
        f.write(f"# overlapping: {len(overlapping_subgraph)}\n")
        f.write(json.dumps(overlapping_subgraph))
    print(f"# rules: {nr_rules}")


if __name__ == "__main__":
    merge()
