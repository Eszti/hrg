import argparse
import itertools
import os

import stanza
from tqdm import tqdm
from tuw_nlp.graph.ud_graph import UDGraph

from source.iwcs.carb.oie_readers.goldReader import GoldReader
from source.iwcs.utils import (
    words_to_idx,
    add_info_to_node,
    contract_triplet_elements,
    OverlappingException,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str)
    parser.add_argument("--out_dir", type=str)
    return parser


def gen_carb_graphs(gold_fn, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gr = GoldReader()
    gr.read(gold_fn)
    gold = gr.oie

    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        tokenize_pretokenized=True,
    )

    unconn_a0 = []
    unconn_a1 = []
    unconn_p = []
    overlapping_subgraph = []
    all_triplets = 0

    # Gen graphs
    for sen_id, (sentence, extractions) in tqdm(enumerate(gold.items())):
        print(f"Processing sen {sen_id}")

        if not sentence:
            continue

        parsed_doc = nlp(sentence)

        for extraction_id, extraction in enumerate(extractions):
            print(f"Processing triplet {extraction_id}")
            all_triplets += 1

            index_dict = {}

            subj = extraction.args[0].strip()
            subj_idx = words_to_idx(subj, parsed_doc.sentences[0])
            if len(subj_idx) == 0:
                unconn_a0.append((sen_id, extraction_id))
            else:
                index_dict["subj"] = subj_idx
            relation = extraction.pred.strip()
            rel_idx = words_to_idx(relation, parsed_doc.sentences[0])
            if len(rel_idx) == 0:
                unconn_p.append((sen_id, extraction_id))
            else:
                index_dict["rel"] = rel_idx
            if len(extraction.args) > 1:
                obj = extraction.args[1].strip()
                obj_idx = words_to_idx(obj, parsed_doc.sentences[0])
                if len(obj_idx) == 0:
                    unconn_a1.append((sen_id, extraction_id))
                else:
                    index_dict["obj"] = obj_idx

            # Save UD graph
            ud_graph = UDGraph(parsed_doc.sentences[0])
            add_info_to_node(ud_graph, index_dict)
            with open(f"{out_dir}/{sen_id}_{extraction_id}.dot", "w") as f:
                f.write(
                    ud_graph.to_dot(
                        marked_nodes=list(itertools.chain(*index_dict.values()))
                    )
                )

            # Contracted graph
            contracted_ud = UDGraph(parsed_doc.sentences[0])
            try:
                heads = contract_triplet_elements(
                    contracted_ud, index_dict, {"subj": "A0", "rel": "P", "obj": "A1"}
                )
            except OverlappingException:
                overlapping_subgraph.append((sen_id, extraction_id))
                continue
            if set(heads.values()) - set(contracted_ud.G):
                overlapping_subgraph.append((sen_id, extraction_id))
                continue

            # Triplet graph
            triplet_graph = contracted_ud.subgraph(
                heads.values(), handle_unconnected="shortest_path"
            ).pos_edge_graph()

            # Save contracted graphs
            add_info_to_node(contracted_ud)
            with open(f"{out_dir}/{sen_id}_{extraction_id}_ud_cont.dot", "w") as f:
                f.write(contracted_ud.to_dot(marked_nodes=heads.values()))
            add_info_to_node(triplet_graph)
            with open(f"{out_dir}/{sen_id}_{extraction_id}_triplet.dot", "w") as f:
                f.write(triplet_graph.to_dot(marked_nodes=heads.values()))

        # if sen_id > 20:
        #     break

    with open(f"{out_dir}/log.txt", "w") as f:
        f.write(f"All triplets: {all_triplets}\n")
        f.write(f"Overlapping subgraph: {len(overlapping_subgraph)}\n")
        f.write(f"{overlapping_subgraph}\n")
        f.write(f"Unconnected A0: {len(unconn_a0)}\n")
        f.write(f"{unconn_a0}\n")
        f.write(f"Unconnected P: {len(unconn_p)}\n")
        f.write(f"{unconn_p}\n")
        f.write(f"Unconnected A1: {len(unconn_a1)}\n")
        f.write(f"{unconn_a1}\n")


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    gen_carb_graphs(args.gold, args.out_dir)
