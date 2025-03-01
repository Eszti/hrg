import logging
from collections import defaultdict

import networkx as nx
from tuw_nlp.graph.graph import Graph

from source.common.rules.hrg_for_triplet import HRGForTriplet
from source.common.rules.hrg_rule import HRGRule
from source.steps.train.train import Train


class TrainComplete(Train):
    def __init__(self, config=None):
        super().__init__(
            description="Script to create hrg rules on preprocessed train data.",
            script_name="train_complete",
            config=config,
        )
        self.method = self.config["method"]
        self.out_dir += f"_{self.method}"

    def _get_hrg_for_triplet(
        self, triplet_graph_str, triplet_id, triplet, triplet_logger
    ):
        triplet_graph = Graph.from_bolinas(triplet_graph_str)
        hrg_for_triplet = None
        if self.method == "per_word":
            hrg_for_triplet = self.__get_rules_per_word(
                triplet_graph, triplet, triplet_id
            )
            for hrg_rule in hrg_for_triplet.all_rules:
                triplet_logger.log(hrg_rule.log_hrg_rule())
        return hrg_for_triplet

    def __get_rules_per_word(self, triplet_graph, triplet, triplet_id):
        hrg_rules = []
        root_word = next(nx.topological_sort(triplet_graph.G))
        next_edges, root_pos = self.__get_next_edges(
            triplet_graph.G, root_word, triplet
        )
        hrg_rules.append(
            self.__get_initial_rule(next_edges, root_pos, triplet, triplet_id)
        )
        for hrg_rule in self.__gen_subseq_rules(
            triplet_graph.G, next_edges, triplet, triplet_id
        ):
            hrg_rules.append(hrg_rule)
        return HRGForTriplet(hrg_rules, triplet_id)

    @staticmethod
    def __get_next_edges(G, root_word, triplet):
        next_edges = defaultdict(list)
        root_pos = ""
        for _, v, e in G.edges(root_word, data=True):
            node_idx = int(v.split("n")[-1])
            label = triplet.get_label(node_idx)
            if label:
                next_edges[label[0]].append((e["color"], v))
            elif node_idx >= 1000:
                root_pos = e["color"]
            else:
                next_edges["X"].append((e["color"], v))
        return next_edges, root_pos

    def __gen_subseq_rules(self, G, pred_edges, triplet, triplet_id):
        for lhs, edges in pred_edges.items():
            for dep_rel, node in edges:
                next_edges, root_pos = self.__get_next_edges(G, node, triplet)

                rule = f"(. :{dep_rel} (."
                for non_term in next_edges:
                    rule += " " + " ".join(
                        f":{non_term}$" for _ in next_edges[non_term]
                    )
                rule += f" :{root_pos} .));\n"
                yield HRGRule(lhs, rule, triplet, triplet_id)

                yield from self.__gen_subseq_rules(G, next_edges, triplet, triplet_id)

    @staticmethod
    def __get_initial_rule(next_edges, root_pos, triplet, triplet_id):
        if len(next_edges) == 0:
            return None
        rule = "(."
        for lhs in sorted(next_edges.keys()):
            rule += " " + " ".join(f":{lhs}$" for _ in next_edges[lhs])
        rule += f" :{root_pos} .);\n"
        return HRGRule("S", rule, triplet, triplet_id)


if __name__ == "__main__":
    logging.getLogger("penman").setLevel(logging.ERROR)
    TrainComplete().run()
