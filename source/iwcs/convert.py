import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str)
    parser.add_argument("--out", type=str)

    return parser


parser = parse_args()
args = parser.parse_args()

in_f = open(args.inp, "r")
out_f = open(args.out, "w")

lines = in_f.readlines()

sen_start = True
sen = ""

for line in lines:
    line = line.strip()
    if sen_start:
        sen = line
        sen_start = False
        continue
    if line == "":
        sen_start = True
        continue

    confidence = float(line.strip().split(":")[0])
    triplet_str = line.strip().split("(")[-1][:-1]
    triplet_elements = triplet_str.split(";")

    if len(triplet_elements) < 2:
        continue
    arg1 = triplet_elements[0]
    relation = triplet_elements[1]
    arg2 = ""
    if len(triplet_elements) > 2:
        srg2 = " ".join(triplet_elements[2:])
    out_f.write(
        f"{sen}\t<arg1> {arg1} </arg1> <rel> {relation} </rel> <arg2> {srg2} </arg2>\t{confidence}\n"
    )

out_f.close()
