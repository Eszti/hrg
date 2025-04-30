import argparse
import itertools
import os
from itertools import product

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
from source.iwcs.utils import add_info_to_node, contract_args, OverlappingException


def words_to_idx(arg, parsed_sen):
    tokens = [w.text for w in parsed_sen.words]
    words = arg.split(" ")
    len_arg = len(words)
    arg_idx = []
    for i in range(len(tokens) - len_arg):
        if " ".join(tokens[i : i + len_arg]) == arg:
            arg_idx.append([j for j in range(i, i + len_arg)])
    if len(arg_idx) == 0:
        all_matches = []
        for word in words:
            new_idx = [j for j, t in enumerate(tokens) if t == word]
            all_matches.append(new_idx)

        # Roll span forwards
        for k in range(1, len(all_matches)):
            if len(all_matches[k]) > 1 and len(all_matches[k - 1]) == 1:
                if all_matches[k - 1][0] + 1 in all_matches[k]:
                    all_matches[k] = [all_matches[k - 1][0] + 1]
        # Roll span backwards
        for k in range(1, len(all_matches)):
            if len(all_matches[-k]) == 1 and len(all_matches[-k - 1]) > 1:
                if all_matches[-k][0] - 1 in all_matches[-k - 1]:
                    all_matches[-k - 1] = [all_matches[-k][0] - 1]

        # Find span in multi-option ranges in between
        for k in range(len(all_matches)):
            if len(all_matches[k]) > 1:
                l = k + 1
                while l < len(all_matches) and len(all_matches[l]) > 1:
                    l += 1
                if l > k + 1:
                    for idx in all_matches[k]:
                        next_idx = idx + 1
                        found = True
                        for m in range(k + 1, l):
                            if next_idx in all_matches[m]:
                                next_idx += 1
                            else:
                                found = False
                                break
                        if found:
                            new_idx = idx
                            for m in range(k, l):
                                assert new_idx in all_matches[m]
                                all_matches[m] = [new_idx]
                                new_idx += 1
                            break

        # Remove uniques from multi-options
        uniques = set()
        multiples = []
        for k in range(len(all_matches)):
            if len(all_matches[k]) == 1:
                uniques.add(all_matches[k][0])
            elif len(all_matches[k]) > 1:
                multiples.append(k)
        for k in multiples:
            new_idx_l = list(set(all_matches[k]) - uniques)
            assert len(new_idx_l) > 0
            all_matches[k] = new_idx_l
            if len(new_idx_l) == 1:
                uniques.add(new_idx_l[0])

        # Take closest in single multi-options - first
        if len(all_matches[0]) > 1 and len(all_matches[1]) == 1:
            sorted_idx = sorted(all_matches[0], reverse=True)
            for idx in sorted_idx:
                if idx < all_matches[1][0]:
                    all_matches[0] = [idx]
                    break
        # Take closest in single multi-options - last
        if len(all_matches[-1]) > 1 and len(all_matches[-2]) == 1:
            sorted_idx = sorted(all_matches[-1])
            for idx in sorted_idx:
                if idx > all_matches[-2][0]:
                    all_matches[-1] = [idx]
                    break
        # Take closest in single multi-options - middle
        for k in range(1, len(all_matches) - 1):
            if (
                len(all_matches[k]) > 1
                and len(all_matches[k - 1]) == 1
                and len(all_matches[k + 1]) == 1
            ):
                for idx in all_matches[k]:
                    if all_matches[k - 1][0] < idx < all_matches[k + 1][0]:
                        all_matches[k] = [idx]

        # Take one-one for duplicates after each other
        for k in range(len(all_matches) - 1):
            if len(all_matches[k]) > 1 and sorted(all_matches[k]) == sorted(
                all_matches[k + 1]
            ):
                sorted_duplicates = sorted(all_matches[k])
                all_matches[k] = [sorted_duplicates[0]]
                all_matches[k + 1] = [sorted_duplicates[1]]

        # Backprop from first single
        first_single = 0
        for k in range(len(all_matches)):
            if len(all_matches[k]) == 1:
                first_single = k
                break
        if first_single > 0:
            highest_idx = all_matches[first_single][0]
            for k in sorted(range(first_single), reverse=True):
                for candidate in sorted(all_matches[k], reverse=True):
                    if candidate < highest_idx:
                        all_matches[k] = [candidate]
                        highest_idx = candidate
                        break

        arg_idx = list(product(*all_matches))
    # Check unique indices
    if len(arg_idx) > 1:
        arg_idx = [arg_idx[0]]
    if len(arg_idx) == 0:
        return []

    assert len(arg_idx) == 1
    arg_idx = [idx + 1 for idx in arg_idx[0]]
    return arg_idx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--gr", type=str)
    return parser


def filter_predictions():
    debug_dir = "filter_debug"
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
                arg_heads = contract_args(
                    contracted_ud, index_dict, {"subj": "A", "rel": "P", "obj": "A"}
                )
            except OverlappingException:
                overlapping_subgraph.append((sen_id, extraction_id))
                validated = add_validated(
                    validated, out_f, sentence, subj, relation, obj, confidence
                )
                continue
            if set(arg_heads.values()) - set(contracted_ud.G):
                overlapping_subgraph.append((sen_id, extraction_id))
                validated = add_validated(
                    validated, out_f, sentence, subj, relation, obj, confidence
                )
                continue

            # Triplet graph
            triplet_graph = contracted_ud.subgraph(
                arg_heads.values(), handle_unconnected="shortest_path"
            ).pos_edge_graph()

            # Bolinas graph
            triplet_graph_str = triplet_graph.to_bolinas(
                keep_node_ids=True, add_n_prefix=True
            )

            # Save graphs
            add_info_to_node(contracted_ud)
            with open(f"{debug_dir}/{sen_id}_{extraction_id}_contracted.dot", "w") as f:
                f.write(contracted_ud.to_dot(marked_nodes=arg_heads.values()))
            add_info_to_node(triplet_graph)
            with open(f"{debug_dir}/{sen_id}_{extraction_id}_triplet.dot", "w") as f:
                f.write(triplet_graph.to_dot(marked_nodes=arg_heads.values()))

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

                validated = add_validated(
                    validated, out_f, sentence, subj, relation, obj, confidence
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


def add_validated(validated, out_f, sentence, subj, relation, obj, confidence):
    validated += 1
    out_f.write(
        f"{sentence}\t<arg1> {subj} </arg1> <rel> {relation} </rel> <arg2> {obj} </arg2>\t{confidence}\n"
    )
    return validated


if __name__ == "__main__":
    filter_predictions()
