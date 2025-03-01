import copy
import re
from collections import Counter

from source.common.derivation.derivation import Derivation
from source.common.triplet.triplet import Triplet


class ProcessedDerivation(Derivation):

    def __init__(self, derivation, score_name=None, pos_tag_resolution=False):
        super().__init__(derivation)
        self.score_name = "derivation_score" if score_name is None else score_name

        self.derived_nodes = sorted(
            list(self.final_item.nodeset),
            key=lambda node: int(node[1:]),
        )
        self.pos_tag_resolution = pos_tag_resolution

        self.rules_counter = Counter()
        self.used_rules = self.__get_rules()

        self.derived_labels = {}
        self.__derive_labels()

        self.original_triplet = Triplet(
            self.derived_labels,
            derivation_score=self.score,
            label_to_nodes=False,
        )
        self.processed_triplet = None

    def calculate_processed_triplet_from_tree_structure(
        self, pos_tags, top_order, score=None
    ):
        self.processed_triplet = copy.copy(self.original_triplet)
        self.processed_triplet.derivation_score = (
            score if score is not None else self.score
        )
        self.processed_triplet.resolve_pred(
            pos_tags, top_order, self.pos_tag_resolution
        )

    def calculate_processed_triplet_from_rule(self, score=None):
        self.processed_triplet = copy.copy(self.original_triplet)
        self.processed_triplet.derivation_score = (
            score if score is not None else self.score
        )
        node_to_order = self.final_item.rule.rhs1.get_nodes()
        order_to_node_id = {o: n for n, o in node_to_order.items()}
        pred_order_ids = [
            order_to_node_id[int(p)] for p in self.final_item.rule.predicates.split("_")
        ]
        pred_node_ids = [self.final_item.mapping[p] for p in pred_order_ids]
        pred_ids = [int(p.split("n")[-1]) for p in pred_node_ids]
        self.processed_triplet.set_pred_ids(pred_ids)

    def full_log(self, logger, k):
        if self.processed_triplet is not None:
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
        logger.log(f"Derived triplet:\n{self.original_triplet.to_short_json()}")
        if self.processed_triplet is not None:
            logger.log(
                f"Processed triplet:\n{self.processed_triplet.to_short_json()}\n"
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
        logger.log(f"Used rules:")
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

    def __derive_labels(self):
        if self.pos_tag_resolution:
            self.__derive_pos_tag_labels()
        else:
            self.arg_counter = -1
            self.__derive_labels_from_nt(self.raw_derivation)

    def __derive_pos_tag_labels(self):
        for u, e, v in self.final_item.shifted:
            label = None
            if re.match(r"A\d\d?", e):
                label = e
            elif re.match(r"[A-Z_]+", e):
                label = "P"
            if label is not None:
                node = u[0].split("n")[-1]
                assert node not in self.derived_labels
                self.derived_labels[node] = label

    def __derive_labels_from_nt(self, derivation, parent_label="S"):
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
                self.__derive_labels_from_nt(child_item, item_label)

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
