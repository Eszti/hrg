import copy
from collections import defaultdict

from source.common.derivation.post_processed_derivation import PostProcessedDerivation
from source.common.derivation.post_processed_derivation_list import (
    PostProcessedDerivationList,
)
from source.common.triplet.triplet_matcher import TripletMatcher


class DerivationList:
    def __init__(self, derivation_list):
        self.derivation_list = derivation_list

    def get_k_best_unique_derivation(self, k, pos_tags, top_order):
        kbest_unique_nodes = set()
        kbest_unique_derivations = []
        for derivation in self.derivation_list:
            post_processed_derivation = PostProcessedDerivation(
                derivation, pos_tags, top_order
            )
            nodes_str = " ".join(post_processed_derivation.derived_nodes)
            if nodes_str not in kbest_unique_nodes:
                kbest_unique_nodes.add(nodes_str)
                kbest_unique_derivations.append(post_processed_derivation)
            if len(kbest_unique_derivations) >= k:
                break
        assert len(kbest_unique_derivations) == len(kbest_unique_nodes)
        if len(kbest_unique_derivations) < k:
            print(f"Found only {len(kbest_unique_derivations)} derivations.")
        return PostProcessedDerivationList(kbest_unique_derivations)

    def get_best_matching_derivations(
        self, gold_triplets, top_order, pos_tags, arg_perm=False
    ):
        derivations_to_keep = defaultdict(lambda: [None] * len(gold_triplets))
        max_scores = defaultdict(lambda: [-1.0] * len(gold_triplets))

        for derivation in self.derivation_list:
            postprocessed_derivation = PostProcessedDerivation(
                derivation, top_order, pos_tags
            )
            original_triplet = postprocessed_derivation.post_processed_triplet

            original_triplet.calculate_all_permutations()
            permutations = (
                [original_triplet] if arg_perm else original_triplet.permutations
            )

            for pred in permutations:
                for i, gold in enumerate(gold_triplets):
                    if not gold.match(pred):
                        continue
                    scores = TripletMatcher(gold, pred).get_scores()
                    for metric, score in scores.items():
                        if score > max_scores[metric][i]:
                            new_derivation = copy.copy(postprocessed_derivation)
                            new_derivation.score = score
                            new_derivation.score_name = metric
                            derivations_to_keep[metric][i] = new_derivation
                            max_scores[metric][i] = score

        ret = defaultdict(list)
        for metric, derivations_for_metric in derivations_to_keep.items():
            for i, derivation in enumerate(derivations_for_metric):
                if derivation is None:
                    assert max_scores[metric][i] == -1.0
                else:
                    ret[metric].append(derivation)
        ret = {
            k: PostProcessedDerivationList(
                sorted(v, key=lambda x: x.score, reverse=True)
            )
            for k, v in ret.items()
        }
        return ret
