import argparse
import json
import os
import sys

import networkx as nx
import stanza

from tuw_nlp.common.vocabulary import Vocabulary
from tuw_nlp.sem.hrg.common.conll import get_sen_from_conll_sen

from tuw_nlp.sem.hrg.common.preproc import get_ud_graph, get_pred_and_args, get_pred_arg_subgraph, add_oie_data_to_nodes
from tuw_nlp.sem.hrg.common.io import create_sen_dir, parse_doc, save_bolinas_str, save_as_dot
from tuw_nlp.text.utils import gen_tsv_sens


def save_conll(sen, fn):
    with open(fn, 'w') as f:
        for line in sen:
            line[0] = str(int(line[0]) + 1)
            f.write("\t".join(line))
            f.write("\n")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--first", type=int)
    parser.add_argument("-l", "--last", type=int)
    parser.add_argument("-o", "--out-dir", type=str)
    return parser.parse_args()


def add_node_labels(bolinas_graph):
    for node, data in bolinas_graph.G.nodes(data=True):
        name = data['name']
        if not name:
            data["name"] = node


def main(out_dir, first=None, last=None):
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        tokenize_pretokenized=True,
    )
    vocab = Vocabulary(first_id=1000)
    last_sen = ""
    preproc_dir = ""
    for sen_idx, sen in enumerate(gen_tsv_sens(sys.stdin)):
        if first is not None and sen_idx < first:
            continue
        if last is not None and last < sen_idx:
            break
        sen_txt = get_sen_from_conll_sen(sen)

        if sen_txt != last_sen:
            sen_dir = create_sen_dir(out_dir, sen_idx)
            preproc_dir = os.path.join(sen_dir, "preproc")
            if not os.path.exists(preproc_dir):
                os.makedirs(preproc_dir)

        log = open(f"{preproc_dir}/sen{sen_idx}.log", "w")
        print(f"processing sentence {sen_idx}, writing to {preproc_dir}/sen{sen_idx}.log")

        save_conll(sen, f"{preproc_dir}/sen{sen_idx}.conll")

        parsed_doc = parse_doc(nlp, sen, preproc_dir, log, save=sen_txt != last_sen)

        args, pred, gold_labels = get_pred_and_args(sen, sen_idx, log)

        ud_graph = get_ud_graph(parsed_doc)

        if sen_txt != last_sen:
            top_order = [n for n in nx.topological_sort(ud_graph.G)]
            json.dump(top_order, open(f"{preproc_dir}/pos_edge_graph_top_order.json", "w"))
            bolinas_graph = ud_graph.pos_edge_graph(vocab)
            save_bolinas_str(f"{preproc_dir}/pos_edge.graph", bolinas_graph, log)
            save_bolinas_str(f"{preproc_dir}/pos_edge_with_labels.graph", bolinas_graph, log, add_names=True)
            add_node_labels(bolinas_graph)
            save_as_dot(f"{preproc_dir}/pos_edge_graph.dot", bolinas_graph, log)

        pred_arg_subgraph = get_pred_arg_subgraph(ud_graph, pred, args, vocab, log)
        save_as_dot(f"{preproc_dir}/sen{sen_idx}_pa_graph.dot", pred_arg_subgraph, log)
        save_bolinas_str(f"{preproc_dir}/sen{sen_idx}_pa.graph", pred_arg_subgraph, log)
        with open(f"{preproc_dir}/sen{sen_idx}_pa_nodes.json", "w") as f:
            json.dump([f"n{n}" for n in pred_arg_subgraph.G.nodes()], f)

        add_oie_data_to_nodes(ud_graph, gold_labels)
        save_as_dot(f"{preproc_dir}/sen{sen_idx}_ud.dot", ud_graph, log)

        with open(f"{preproc_dir}/sen{sen_idx}_gold_labels.json", "w") as f:
            json.dump(gold_labels, f)

        last_sen = sen_txt


if __name__ == "__main__":
    args = get_args()
    main(args.out_dir, args.first, args.last)
