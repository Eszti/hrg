import argparse
import json
import re
from collections import Counter


def merge(in_dir, id_fn, out):
    ids = json.load(open(id_fn))

    lines = ["TP/FP anal\n"]
    subj_tags = Counter()
    obj_tags = Counter()

    for idx in ids:
        id_str = f"{idx[0]}_{idx[1]}"
        lines.append(f"\n{id_str}\n")
        with open(f"{in_dir}/{id_str}.txt") as f:
            sen_lines = f.readlines()
            sen = sen_lines[0]
            predicted = sen_lines[1]
            lines.append(f"{sen}")
            lines.append(f"{predicted}\n")
            lines.append("\n")
        with open(f"{in_dir}/{id_str}_np_chunks.txt") as f:
            lines.extend(f.readlines())
            lines.append("\n\n")
        with open(f"{in_dir}/{id_str}_sen_const.txt") as f:
            sen_const = f.readlines()[0]
            lines.append(f"{sen_const}\n")
            subj = predicted.split("\t")[1].strip()
            obj = predicted.split("\t")[2].strip()

            subj_start = subj.split(" ")[0]
            subj_tag_search = re.search(
                f"\(([A-Z]+) \([A-Z]+ {subj_start}\)", sen_const
            )
            if subj_tag_search:
                subj_tag = subj_tag_search.group(1)
                subj_tags[subj_tag] += 1
                lines.append(f"\nsubj tag: {subj_tag}\n")

            obj_start = obj.split(" ")[0]
            obj_tag_search = re.search(f"\(([A-Z]+) \([A-Z]+ {obj_start}\)", sen_const)
            if obj_tag_search:
                obj_tag = obj_tag_search.group(1)
                obj_tags[obj_tag] += 1
                lines.append(f"obj tag: {obj_tag}\n")

    with open(f"{out}", "w") as f:
        f.writelines(lines)

    print(subj_tags)
    print(obj_tags)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str)
    parser.add_argument("--id-fn", type=str)
    parser.add_argument("--out", type=str)
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    merge(args.in_dir, args.id_fn, args.out)
