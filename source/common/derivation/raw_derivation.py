import re

from common.bolinas.hgraph import Hgraph


class RawDerivation:

    def __init__(self, raw_derivation):
        self.raw_derivation = raw_derivation[1]
        self.score = raw_derivation[0]

    def _log_raw_derivation(self, logger, k):
        logger.log(f"K{k}")
        logger.log(f"Raw score: {self.score:g}")
        logger.log(self.print_shifted())
        logger.log(f"{self.__format_derivation()}\n")

    def print_shifted(self):
        final_item = self.raw_derivation[1]["START"][0]
        node_to_concepts = dict(zip(final_item.nodeset, [""] * len(final_item.nodeset)))
        triples = []
        for v, l, u in final_item.shifted:
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
                    (rel, RawDerivation.walk_derivation(c, combiner, leaf))
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

        return RawDerivation.walk_derivation(self.raw_derivation, combiner, leaf)
