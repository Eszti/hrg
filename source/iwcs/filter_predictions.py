import argparse
import itertools
import os

import stanza
from tqdm import tqdm
from tuw_nlp.graph.ud_graph import UDGraph

from source.common.bolinas.grammar import Grammar
from source.common.bolinas.parser import Parser
from source.common.bolinas.vo_rule import VoRule
from source.common.exceptions import (
    ParseTooLongException,
    CkyTooLongException,
    NotAllNodesCoveredException,
)
from source.iwcs.carb.oie_readers.allennlpReader import AllennlpReader
from source.iwcs.utils import (
    add_info_to_node,
    contract_triplet_elements,
    OverlappingException,
    words_to_idx,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--gr", type=str)
    return parser


def filter_predictions():
    debug_dir = "filter_openie6_debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    parser = parse_args()
    args = parser.parse_args()

    out_f = open(args.out, "w")
    log_f = open(f"filtered.log", "w")
    sentences = set()
    predictions = set()

    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        tokenize_pretokenized=True,
    )

    with open(args.gr) as f:
        grammar = Grammar.load_from_file(
            f, VoRule, reverse=False, nodelabels=True, logprob=True
        )

    parser = Parser(grammar, stop_at_first=True, permutations=False)

    # Index mapping errors
    arg2plus = []
    unconn_span = []

    # Triplet contraction
    overlapping_subgraph = []

    # Validation errors
    parse_error = []
    cky_error = []
    deriv_error = []

    # Filter stat
    all_triplets = 0
    validated = 0
    not_validated = []

    predicted = AllennlpReader(threshold=None)
    predicted.read(args.inp)

    for sen_id, (sentence, extractions) in tqdm(enumerate(predicted.oie.items())):
        print(f"Processing sen {sen_id}")

        if not sentence:
            continue

        sentences.add(sentence)
        parsed_doc = nlp(sentence)

        for extraction_id, extraction in enumerate(extractions):
            if extraction in predictions:
                pass
            print(f"Processing triplet {extraction_id}")
            all_triplets += 1

            with open(f"{debug_dir}/{sen_id}_{extraction_id}.txt", "w") as f:
                f.write(f"{sentence}\n")
                f.write(str(extraction))

            predictions.add(extraction)
            confidence = extraction.confidence
            if len(extraction.args) > 2:
                arg2plus.append((sentence, extraction_id))
                # Todo
                continue

            index_dict = {}

            subj, relation, obj = "", "", ""

            subj = extraction.args[0].strip()
            subj_idx = words_to_idx(subj, parsed_doc.sentences[0])
            if len(subj_idx) == 0:
                unconn_span.append((sen_id, extraction_id))
            else:
                index_dict["subj"] = subj_idx
            relation = extraction.pred.strip()
            rel_idx = words_to_idx(relation, parsed_doc.sentences[0])
            if len(rel_idx) == 0:
                unconn_span.append((sen_id, extraction_id))
            else:
                index_dict["rel"] = rel_idx
            if len(extraction.args) > 1:
                obj = extraction.args[1].strip()
                obj_idx = words_to_idx(obj, parsed_doc.sentences[0])
                if len(obj_idx) == 0:
                    unconn_span.append((sen_id, extraction_id))
                else:
                    index_dict["obj"] = obj_idx

            # Save UD graph
            ud_graph = UDGraph(parsed_doc.sentences[0])
            add_info_to_node(ud_graph, index_dict)
            with open(f"{debug_dir}/{sen_id}_{extraction_id}.dot", "w") as f:
                f.write(
                    ud_graph.to_dot(
                        marked_nodes=list(itertools.chain(*index_dict.values()))
                    )
                )

            # Contracted graph
            contracted_ud = UDGraph(parsed_doc.sentences[0])
            try:
                heads = contract_triplet_elements(
                    contracted_ud, index_dict, {"subj": "A", "rel": "P", "obj": "A"}
                )
            except OverlappingException:
                overlapping_subgraph.append((sen_id, extraction_id))
                # validated = add_validated(
                #     validated, out_f, sentence, subj, relation, obj, confidence
                # )
                continue
            if set(heads.values()) - set(contracted_ud.G):
                overlapping_subgraph.append((sen_id, extraction_id))
                # validated = add_validated(
                #     validated, out_f, sentence, subj, relation, obj, confidence
                # )
                continue

            # Triplet graph
            triplet_graph = contracted_ud.subgraph(
                heads.values(), handle_unconnected="shortest_path"
            ).pos_edge_graph()

            # Bolinas graph
            triplet_graph_str = triplet_graph.to_bolinas(
                keep_node_ids=True, add_n_prefix=True
            )

            # Save graphs
            add_info_to_node(contracted_ud)
            with open(f"{debug_dir}/{sen_id}_{extraction_id}_contracted.dot", "w") as f:
                f.write(contracted_ud.to_dot(marked_nodes=heads.values()))
            add_info_to_node(triplet_graph)
            with open(f"{debug_dir}/{sen_id}_{extraction_id}_triplet.dot", "w") as f:
                f.write(triplet_graph.to_dot(marked_nodes=heads.values()))

            # Validate
            try:
                derivation = parser.check_membership(
                    triplet_graph_str,
                    sen_logger=None,
                    global_logger=None,
                    pos_tag_resolution=True,
                )

                if derivation is None:
                    not_validated.append((sen_id, extraction_id))
                    continue

                assert len(derivation.used_rules.values()) == 1
                r_id = list(derivation.used_rules.keys())[0]
                rule = list(derivation.used_rules.values())[0]
                rule_str = f"{r_id}: {rule}"
                validated = add_validated(
                    validated,
                    out_f,
                    sentence,
                    subj,
                    relation,
                    obj,
                    confidence,
                    rule_str,
                    f"{sen_id}_{extraction_id}",
                )
            except ParseTooLongException as e:
                parse_error.append((sen_id, extraction_id))
            except CkyTooLongException as e:
                cky_error.append((sen_id, extraction_id))
            except NotAllNodesCoveredException as e:
                deriv_error.append((sen_id, extraction_id))

        # if sen_id > 20:
        #     break

    out_f.close()
    log_f.write(f"All triplets: {all_triplets}\n")
    log_f.write(f"Validated triplets: {validated}\n")
    log_f.write(f"Not validated: {len(not_validated)}\n")
    log_f.write(f"{not_validated}\n")
    log_f.write(f"Parse error: {len(parse_error)}\n")
    log_f.write(f"{parse_error}\n")
    log_f.write(f"CKY error: {len(cky_error)}\n")
    log_f.write(f"{cky_error}\n")
    log_f.write(f"Deriv error: {len(deriv_error)}\n")
    log_f.write(f"{deriv_error}\n")
    log_f.write(f"Overlapping subgraph: {len(overlapping_subgraph)}\n")
    log_f.write(f"{overlapping_subgraph}\n")
    log_f.write(f"Unconnected span: {len(unconn_span)}\n")
    log_f.write(f"{unconn_span}\n")
    log_f.write(f"Arg2+: {len(arg2plus)}\n")
    log_f.write(f"{arg2plus}\n")


def add_validated(
    validated, out_f, sentence, subj, relation, obj, confidence, rule, id_str
):
    validated += 1
    out_f.write(
        f"{sentence}\t"
        f"<arg1> {subj} </arg1> <rel> {relation} </rel> <arg2> {obj} </arg2> <id> {id_str} </id> <rule> {rule} </rule>\t"
        f"{confidence}\n"
    )
    return validated


if __name__ == "__main__":
    filter_predictions()
