import argparse
import itertools
import json
import os.path
import re
from collections import defaultdict, Counter

import stanza
from tuw_nlp.graph.ud_graph import UDGraph
from tuw_nlp.text.utils import gen_tsv_sens

from source.iwcs.utils import add_info_to_node, contract_args, OverlappingException


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--debug_out", type=str, default="train_out")
    return parser.parse_args()


def train(inp, out):
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        tokenize_pretokenized=True,
    )

    if not os.path.exists("train_out"):
        os.makedirs("train_out")

    rules = []
    overlapping_subgraph = []

    # Create rules
    for sen_idx, sen in enumerate(gen_tsv_sens(open(inp))):
        print(f"Processing sen {sen_idx}")
        parsed_doc = nlp(" ".join(t[1] for t in sen))
        # Indices
        index_dict = defaultdict(list)
        for i, tok in enumerate(sen):
            label = tok[7].split("-")[0]
            if label == "P" or label.startswith("A"):
                index_dict[label].append(i + 1)
        ud_graph = UDGraph(parsed_doc.sentences[0])
        add_info_to_node(ud_graph, index_dict)
        with open(f"train_out/{sen_idx}_ud.dot", "w") as f:
            f.write(
                ud_graph.to_dot(
                    marked_nodes=list(itertools.chain(*index_dict.values()))
                )
            )

        # Contract
        contracted_ud = UDGraph(parsed_doc.sentences[0])
        try:
            arg_heads = contract_args(
                contracted_ud,
                index_dict,
                {arg_name: arg_name[0] for arg_name in index_dict},
            )
        except OverlappingException:
            overlapping_subgraph.append(sen_idx)
            continue
        if set(arg_heads.values()) - set(contracted_ud.G):
            overlapping_subgraph.append(sen_idx)
            continue

        # Triplet graph
        triplet_graph = contracted_ud.subgraph(
            arg_heads.values(), handle_unconnected="shortest_path"
        ).pos_edge_graph()

        # Save graphs
        add_info_to_node(contracted_ud)
        with open(f"train_out/{sen_idx}_ud_cont.dot", "w") as f:
            f.write(contracted_ud.to_dot(marked_nodes=arg_heads.values()))
        add_info_to_node(triplet_graph)
        with open(f"train_out/{sen_idx}_triplet.dot", "w") as f:
            f.write(triplet_graph.to_dot(marked_nodes=arg_heads.values()))

        # Create rules
        rhs = triplet_graph.to_bolinas(keep_node_ids=False, add_n_prefix=True)
        rule = f"S -> {rhs}"
        with open(f"train_out/{sen_idx}.hrg", "w") as f:
            f.write(rule)
        rules.append(rule)

    # Write grammar
    c = Counter(rules)
    weighted_rules = [
        (rule, float(c[rule]) / len(rules)) for rule, cnt in c.most_common()
    ]
    with open(out, "w") as f:
        for rule, w in weighted_rules:
            f.write(f"{rule};\t{w}\n")
    with open("train_out/overlapping.txt", "w") as f:
        f.write(f"# overlapping: {len(overlapping_subgraph)}\n")
        f.write(json.dumps(overlapping_subgraph))


if __name__ == "__main__":
    args = parse_args()
    train(args.inp, args.out)
