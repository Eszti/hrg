from collections import Counter

from common.derivation.raw_derivation import RawDerivation
from common.triplet.triplet import Triplet


class ProcessedDerivation(RawDerivation):

    def __init__(self, raw_derivation, score=None, score_name=None):
        super().__init__(raw_derivation)
        self.score = self.score if score is None else score
        self.score_name = "raw_score" if score_name is None else score_name

        self.derived_nodes = sorted(
            list(self.raw_derivation[1]["START"][0].nodeset),
            key=lambda node: int(node[1:]),
        )

        self.rules_counter = Counter()
        self.used_rules = self.__get_rules()

        self.arg_counter = -1
        self.derived_labels = {}
        self.__derive_labels(self.raw_derivation)

        self.raw_triplet = Triplet(self.derived_labels, label_to_nodes=False)

    def full_log(self, logger, k):
        self._log_raw_derivation(logger, k)
        self.__log_used_rules(logger)
        self.__log_derived_nodes(logger, k)

    def __log_derived_nodes(self, logger, k):
        logger.log(f"k{k}:\t{self.derived_nodes} - {len(self.derived_nodes)}\n")

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

        return RawDerivation.walk_derivation(self.raw_derivation, combiner, leaf)

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
