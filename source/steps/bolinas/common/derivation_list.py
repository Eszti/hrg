import copy
from collections import defaultdict

from common.triplet import Triplet
from common.triplet_matcher import TripletMatcher
from steps.bolinas.common.derivation import Derivation
from steps.bolinas.common.exceptions import ScoreDisorderException
from steps.postproc.postproc import postprocess


class DerivationList:
    def __init__(self, derivation_list, raw=False):
        if raw:
            self.derivation_list = [Derivation(derivation=x[1], score=x[0]) for x in derivation_list]
        else:
            self.derivation_list = derivation_list

    def __iter__(self):
        return (x for x in list.__iter__(self.derivation_list))

    def __getitem__(self, item):
        return self.derivation_list.__getitem__(item)

    def get_k_best_unique_derivation(self, k):
        kbest_unique_nodes = set()
        kbest_unique_derivations = []
        for derivation in self.derivation_list:
            derivation.collect_derived_nodes()
            nodes_str = " ".join(derivation.derived_nodes)
            if nodes_str not in kbest_unique_nodes:
                kbest_unique_nodes.add(nodes_str)
                kbest_unique_derivations.append(derivation)
            if len(kbest_unique_derivations) >= k:
                break
        assert len(kbest_unique_derivations) == len(kbest_unique_nodes)
        if len(kbest_unique_derivations) < k:
            print(f"Found only {len(kbest_unique_derivations)} derivations.")
        return DerivationList(kbest_unique_derivations)

    def get_best_matching_derivations(self, gold_triplets, pos_tags, top_order, arg_perm=False):
        derivations_to_keep = defaultdict(lambda: [None] * len(gold_triplets))
        max_scores = defaultdict(lambda: [-1.0] * len(gold_triplets))

        for derivation in self.derivation_list:
            derivation.predict_labels()
            _, all_permutations = postprocess(derivation.predicted_labels, pos_tags, top_order, arg_perm)

            for permutation in all_permutations:
                pred = Triplet(permutation, label_to_nodes=False)
                for i, gold in enumerate(gold_triplets):
                    if not gold.match(pred):
                        continue
                    scores = TripletMatcher(gold, pred).get_scores()
                    for metric, score in scores.items():
                        if score > max_scores[metric][i]:
                            new_derivation = copy.copy(derivation)
                            new_derivation.set_postprocessed_labels(permutation)
                            new_derivation.score = score
                            derivations_to_keep[metric][i] = new_derivation
                            max_scores[metric][i] = score

        ret = defaultdict(list)
        for metric, derivations_for_metric in derivations_to_keep.items():
            for i, derivation in enumerate(derivations_for_metric):
                if derivation is None:
                    assert max_scores[metric][i] == -1.0
                else:
                    ret[metric].append(derivation)
        ret = {k: DerivationList(sorted(v, key=lambda x: x.score, reverse=True)) for k, v
               in ret.items()}
        return ret

    def save_derivations_as_graph_file(self, fn):
        with open(fn, "w") as f:
            for derivation in self.derivation_list:
                f.write(f"{derivation.print_shifted()};{derivation.score:g}\n")

    def save_triplets(self, fn):
        with open(fn, "w") as f:
            for derivation in self.derivation_list:
                derivation.predict_labels()
                f.write(
                    f"{Triplet(derivation.predicted_labels, label_to_nodes=False).to_json_str()};{derivation.score:g}\n"
                )

    def check_score_disorder(self):
        for i in range(len(self.derivation_list)-1):
            score_a = self.derivation_list[i].score
            score_b = self.derivation_list[i+1].score
            if score_a < score_b:
                raise ScoreDisorderException(i, score_a, score_b)

    def log_all_derivations(self, logger):
        logger.log("=====================\n")
        for i, derivation in enumerate(self.derivation_list):
            derivation.full_log(logger, i + 1)
            logger.log("=====================\n")
