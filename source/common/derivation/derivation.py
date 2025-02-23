import re

from source.common.bolinas.hgraph import Hgraph


class Derivation:

    def __init__(self, raw_derivation):
        self.raw_derivation = raw_derivation[1]
        self.score = raw_derivation[0]
        if type(self.raw_derivation[1]["START"]) is tuple:
            self.final_item = self.raw_derivation[1]["START"][0]
        else:
            self.final_item = self.raw_derivation[1]["START"]

    def _log_derivation(self, logger):
        logger.log("Shifted derivation:")
        logger.log(self.__print_shifted())
        logger.log("\nFormat derivation:")
        logger.log(f"{self.__format_derivation()}\n")

    def __print_shifted(self):
        node_to_concepts = dict(
            zip(self.final_item.nodeset, [""] * len(self.final_item.nodeset))
        )
        triples = []
        for v, l, u in self.final_item.shifted:
            triples.append((v[0], l, u[0][0]))
        graph = Hgraph.from_triples(triples, node_to_concepts)
        return re.sub(r"(\n|\s+)", " ", graph.to_bolinas_str(nodeids=True))

    @staticmethod
    def walk_derivation(derivation, combiner, leaf):
        if type(derivation) is not tuple:
            if derivation == "START":
                return None
            return leaf(derivation)
        else:
            item, children = derivation[0], derivation[1]
            childobjs = dict(
                [
                    (rel, Derivation.walk_derivation(c, combiner, leaf))
                    for (rel, c) in children.items()
                ]
            )

            if item == "START":
                return childobjs["START"]

            return combiner(item, childobjs)

    def __format_derivation(self):
        def combiner(item, childobjs):
            children = []
            for nt, child in childobjs.items():
                edgestring = "$".join(nt)
                children.append(f"{edgestring}({child})")
            childstr = " ".join(children)
            return f"{item.rule.rule_id}({childstr})"

        def leaf(item):
            return str(item.rule.rule_id)

        return Derivation.walk_derivation(self.raw_derivation, combiner, leaf)
