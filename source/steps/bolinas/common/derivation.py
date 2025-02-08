import json
import re
from collections import Counter, OrderedDict

from steps.bolinas.common.hgraph.hgraph import Hgraph


class Derivation:

    def __init__(self, derivation, score):
        self.derivation = derivation
        self.score = score
        self.original_score = score
        self.derived_nodes = None
        self.used_rules = None
        self.rules_counter = Counter()
        self.predicted_labels = None
        self.postprocessed_labels = None

    def collect_derived_nodes(self):
        if self.derived_nodes is None:
            final_item = self.derivation[1]["START"][0]
            self.derived_nodes = sorted(
                list(final_item.nodeset), key=lambda node: int(node[1:])
            )

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

        return Derivation.__walk_derivation(self.derivation, combiner, leaf)

    def collect_used_rules(self):
        if self.used_rules is None:
            self.used_rules = self.__get_rules()

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

    def set_postprocessed_labels(self, pp_labels):
        self.postprocessed_labels = pp_labels

    def predict_labels(self):
        if self.predicted_labels is None:
            self.predicted_labels = self.__get_labels()

    def print_predicted_labels(self):
        return f"{json.dumps(OrderedDict(sorted(self.predicted_labels.items(), key=lambda x: int(x[0]))))}"

    def __get_labels(self):
        def combiner(item, childobjs):
            children = leaf(item)
            for nt, child in childobjs.items():
                for k, v in child.items():
                    assert k not in children or v == children[k]
                    children[k] = v
            return children

        def leaf(item):
            nt = item.rule.symbol
            if nt == "S":
                return {}
            return {(item.mapping["_1"].split("n")[1]): nt}

        return Derivation.__walk_derivation(self.derivation, combiner, leaf)

    def __log_derivation(self, logger, k):
        logger.log(f"K{k}")
        logger.log(f"Score: {self.score:g} (Original score: {self.original_score:g})")
        logger.log(self.print_shifted())
        logger.log(f"{self.format_derivation()}\n")

    def print_shifted(self):
        final_item = self.derivation[1]["START"][0]
        node_to_concepts = dict(zip(final_item.nodeset, [""] * len(final_item.nodeset)))
        triples = []
        for v, l, u in final_item.shifted:
            triples.append((v[0], l, u[0][0]))
        graph = Hgraph.from_triples(triples, node_to_concepts)
        return re.sub(r"(\n|\s+)", " ", graph.to_bolinas_str(nodeids=True))

    @staticmethod
    def __walk_derivation(derivation, combiner, leaf):
        if type(derivation) is not tuple:
            if derivation == "START":
                return None
            return leaf(derivation)
        else:
            item, children = derivation[0], derivation[1]
            childobjs = dict(
                [
                    (rel, Derivation.__walk_derivation(c, combiner, leaf))
                    for (rel, c) in children.items()
                ]
            )

            if item == "START":
                return childobjs["START"]

            return combiner(item, childobjs)

    def format_derivation(self):
        def combiner(item, childobjs):
            children = []
            for nt, child in childobjs.items():
                edgestring = "$".join(nt)
                children.append(f"{edgestring}({child})")
            childstr = " ".join(children)
            return f"{item.rule.rule_id}({childstr})"

        def leaf(item):
            return str(item.rule.rule_id)

        return Derivation.__walk_derivation(self.derivation, combiner, leaf)

    def full_log(self, logger, k):
        self.collect_derived_nodes()
        self.collect_used_rules()
        self.__log_derivation(logger, k)
        self.__log_used_rules(logger)
        self.__log_derived_nodes(logger, k)
