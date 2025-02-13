import copy
from collections import Counter

from source.common.derivation.derivation import Derivation
from source.common.triplet.triplet import Triplet


class ProcessedDerivation(Derivation):

    def __init__(self, derivation, score_name=None):
        super().__init__(derivation)
        self.score_name = "derivation_score" if score_name is None else score_name

        self.derived_nodes = sorted(
            list(self.raw_derivation[1]["START"][0].nodeset),
            key=lambda node: int(node[1:]),
        )

        self.rules_counter = Counter()
        self.used_rules = self.__get_rules()

        self.arg_counter = -1
        self.derived_labels = {}
        self.__derive_labels(self.raw_derivation)

        self.original_triplet = Triplet(
            self.derived_labels,
            derivation_score=self.score,
            label_to_nodes=False,
        )
        self.processed_triplet = None

    def calculate_processed_triplet(self, pos_tags, top_order, score=None):
        self.processed_triplet = copy.copy(self.original_triplet)
        self.processed_triplet.derivation_score = (
            score if score is not None else self.score
        )
        self.processed_triplet.resolve_pred(pos_tags, top_order)

    def full_log(self, logger, k):
        logger.log(
            f"K{k}\n{self.score_name} score: {self.processed_triplet.derivation_score:g}"
        )
        logger.log(f"raw derivation score: {self.score:g}\n")
        self._log_derivation(logger)
        self.__log_used_rules(logger)
        self.__log_derived_nodes(logger, k)
        self.__log_triplet(logger)

    def __log_derived_nodes(self, logger, k):
        logger.log(f"k{k}:\t{self.derived_nodes} - {len(self.derived_nodes)}\n")

    def __log_triplet(self, logger):
        logger.log(
            f"Derived triplet:\n{self.original_triplet.to_short_json()} - # nodes: {self.original_triplet.len()}"
        )
        logger.log(
            f"Processed triplet:\n{self.processed_triplet.to_short_json()} - # nodes: {self.processed_triplet.len()}\n"
        )

    def __get_rules(self):
        def combiner(item, childobjs):
            children = leaf(item)
            for nt, child in childobjs.items():
                for k, v in child.items():
                    assert k not in children or v == children[k]
                    children[k] = v
            return children

        def leaf(item):
            rule_id = item.rule.rule_id
            self.rules_counter[rule_id] += 1
            return {rule_id: str(item.rule)}

        return Derivation.walk_derivation(self.raw_derivation, combiner, leaf)

    def __log_used_rules(self, logger):
        for rule_id in sorted(self.used_rules):
            rule_str = self.used_rules[rule_id]
            prob = rule_str.split(";")[1].strip()
            if not prob:
                prob = 0
            rule = rule_str.split(";")[0].strip()
            logger.log(f"{rule_id}\t{round(float(prob), 2)}\t{rule}")
        logger.log(f"\nUsed rules for derivation: {sorted(self.rules_counter.items())}")
        logger.log(f"Number of different used rules: {len(self.rules_counter.keys())}")
        logger.log(f"Total number of used rules: {sum(self.rules_counter.values())}\n")

    def __derive_labels(self, derivation, parent_label="S"):
        if type(derivation) is not tuple:
            self.__add_label(derivation, parent_label)
        else:
            item = derivation[0]
            item_label = self.__add_label(item, parent_label)
            items = sorted(
                [c for (_, c) in derivation[1].items()],
                key=lambda x: self.__child_item_sort_criteria(x),
            )
            for child_item in items:
                self.__derive_labels(child_item, item_label)

    @staticmethod
    def __child_item_sort_criteria(child_item):
        if type(child_item) is not tuple:
            item = child_item
        else:
            item = child_item[0]
        return int(item.mapping["_1"].split("n")[1])

    def __add_label(self, item, parent_label):
        item_label = ""
        if item != "START":
            item_label = item.rule.symbol
        if not parent_label.startswith("A") and item_label == "A":
            self.arg_counter += 1
        if item_label.startswith("P"):
            self.derived_labels[item.mapping["_1"].split("n")[1]] = item_label
        elif item_label.startswith("A"):
            self.derived_labels[item.mapping["_1"].split("n")[1]] = (
                f"A{self.arg_counter}"
            )
        return item_label
