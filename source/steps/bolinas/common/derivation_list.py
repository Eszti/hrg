from steps.bolinas.common.derivation import Derivation


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

    def save_derivations_as_graph_file(self, fn):
        with open(fn, "w") as f:
            for derivation in self.derivation_list:
                f.write(f"{derivation.print_shifted()};{derivation.score:g}\n")

    def save_predicted_labels(self, fn):
        with open(fn, "w") as f:
            for derivation in self.derivation_list:
                derivation.predict_labels()
                f.write(f"{derivation.print_predicted_labels()};{derivation.score:g}\n")

    def check_score_disorder(self, logger):
        cnt = 0
        for i in range(len(self.derivation_list)-1):
            score_a = self.derivation_list[i].score
            score_b = self.derivation_list[i+1].score
            if score_a < score_b:
                logger.log(f"score disorder - {i}: {score_a:g} / {score_b:g}")
                cnt += 1
        return cnt

    def log_all_derivations(self, logger):
        logger.log("=====================\n")
        for i, derivation in enumerate(self.derivation_list):
            derivation.full_log(logger, i+1)
            logger.log("=====================\n")