import argparse
from collections import defaultdict

from tuw_nlp.text.utils import gen_tsv_sens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", nargs="+", type=str)
    parser.add_argument("--sen_out", type=str)
    parser.add_argument("--gold_out", type=str)
    return parser


def merge(inp, sen_out, gold_out):
    sens = list()
    gold = list()
    tuples = 0

    for fn in inp:
        for sen_idx, sen in enumerate(gen_tsv_sens(open(fn))):
            sen_txt = " ".join([line[1] for line in sen])

            p = []
            args = defaultdict(list)
            for i, tok in enumerate(sen):
                label = tok[7].split("-")[0]
                if label == "P":
                    p.append(i + 1)
                elif label.startswith("A"):
                    args[label[1:]].append(i + 1)

            if p and len(args.keys()) >= 2:
                if sen_txt not in sens:
                    sens.append(sen_txt)
                tuples += 1
                pred_txt = " ".join([sen[i - 1][1] for i in p])
                arg_txt = " ".join([sen[i - 1][1] for i in args["0"]]) + "\t"
                arg_txt += " ".join(
                    [
                        " ".join(sen[i - 1][1] for i in args[l])
                        for l in sorted(args.keys(), key=lambda x: int(x))[1:]
                    ]
                )
                gold_line = f"{sen_txt}\t{pred_txt}\t{arg_txt}\n"
                if gold_line not in gold:
                    gold.append(gold_line)

    print(f"# sentences: {len(sens)}")
    print(f"# tuples: {tuples}")

    with open(sen_out, "w") as f:
        for sen_txt in sens:
            f.write(sen_txt + "\n")

    with open(gold_out, "w") as f:
        f.writelines(gold)


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    merge(args.inp, args.sen_out, args.gold_out)
