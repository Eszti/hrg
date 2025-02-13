import copy
from collections import defaultdict

from source.common.derivation.kbest_derivation_list import (
    KbestDerivationList,
)
from source.common.derivation.processed_derivation import ProcessedDerivation
from source.common.triplet.triplet_matcher import TripletMatcher


class DerivationList:
    def __init__(self, derivation_list):
        self.derivation_list = derivation_list

    def get_k_best_unique_derivation(self, sen_id, sen_text, k, pos_tags, top_order):
        kbest_unique_nodes = set()
        kbest_unique_derivations = []
        for derivation in self.derivation_list:
            processed_derivation = ProcessedDerivation(derivation)
            nodes_str = " ".join(processed_derivation.derived_nodes)
            if nodes_str not in kbest_unique_nodes:
                kbest_unique_nodes.add(nodes_str)
                processed_derivation.calculate_processed_triplet(pos_tags, top_order)
                processed_derivation.processed_triplet.triplet_id = (
                    len(kbest_unique_derivations) + 1
                )
                kbest_unique_derivations.append(processed_derivation)
            if len(kbest_unique_derivations) >= k:
                break
        assert len(kbest_unique_derivations) == len(kbest_unique_nodes)
        if len(kbest_unique_derivations) < k:
            print(f"Found only {len(kbest_unique_derivations)} derivations.")
        return KbestDerivationList(kbest_unique_derivations, sen_id, sen_text)

    def get_best_matching_derivations(
        self, sen_id, sen_text, gold_triplets, top_order, pos_tags, arg_perm=True
    ):
        derivations_to_keep = defaultdict(lambda: [None] * len(gold_triplets))
        max_scores = defaultdict(lambda: [-1.0] * len(gold_triplets))

        for derivation in self.derivation_list:
            processed_derivation = ProcessedDerivation(derivation)
            processed_derivation.calculate_processed_triplet(pos_tags, top_order)
            permutations = (
                processed_derivation.processed_triplet.get_all_permutations()
                if arg_perm
                else [processed_derivation.processed_triplet]
            )

            for pred in permutations:
                for i, gold in enumerate(gold_triplets):
                    if not gold.match(pred):
                        continue
                    scores = TripletMatcher(gold, pred).get_scores()
                    for metric, score in scores.items():
                        if score > max_scores[metric][i]:
                            new_derivation = copy.copy(processed_derivation)
                            new_derivation.processed_triplet = copy.copy(pred)
                            new_derivation.processed_triplet.derivation_score = score
                            new_derivation.score_name = metric
                            derivations_to_keep[metric][i] = new_derivation
                            max_scores[metric][i] = score

        ret = defaultdict(list)
        for metric, derivations_for_metric in derivations_to_keep.items():
            for i, derivation in enumerate(derivations_for_metric):
                if derivation is None:
                    assert max_scores[metric][i] == -1.0
                else:
                    derivation.processed_triplet.triplet_id = len(ret[metric]) + 1
                    ret[metric].append(derivation)
        ret = {
            k: KbestDerivationList(
                sorted(v, key=lambda x: x.score, reverse=True),
                sen_id=sen_id,
                sen_text=sen_text,
            )
            for k, v in ret.items()
        }
        return ret
