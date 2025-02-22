import logging
from collections import defaultdict

import networkx as nx

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

    def _get_rules(self, triplet_graph, triplet):
        rules_dict = {}
        if self.method == "per_word":
            initial_rule, rules = self.__get_rules_per_word(triplet_graph, triplet)
            rules_dict["initial_rule"] = initial_rule
            rules_dict["rules"] = rules
        return rules_dict

    def __get_rules_per_word(self, triplet_graph, triplet):
        root_word = next(nx.topological_sort(triplet_graph.G))
        next_edges, root_pos = self.__get_next_edges(
            triplet_graph.G, root_word, triplet
        )
        initial_rule = self.__get_initial_rule(next_edges, root_pos)
        rules = set()
        for rule in self.__gen_subseq_rules(triplet_graph.G, next_edges, triplet):
            rules.add(rule)
        return initial_rule, rules

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

    def __gen_subseq_rules(self, G, pred_edges, triplet):
        for lhs, edges in pred_edges.items():
            for dep_rel, node in edges:
                next_edges, root_pos = self.__get_next_edges(G, node, triplet)

                rule = f"{lhs} -> (. :{dep_rel} (."
                for non_term in next_edges:
                    rule += " " + " ".join(
                        f":{non_term}$" for _ in next_edges[non_term]
                    )
                rule += f" :{root_pos} .));\n"
                yield rule

                yield from self.__gen_subseq_rules(G, next_edges, triplet)

    @staticmethod
    def __get_initial_rule(next_edges, root_pos):
        if len(next_edges) == 0:
            return None
        rule = "S -> (."
        for lhs in sorted(next_edges.keys()):
            rule += " " + " ".join(f":{lhs}$" for _ in next_edges[lhs])
        rule += f" :{root_pos} .);\n"
        return rule


if __name__ == "__main__":
    logging.getLogger("penman").setLevel(logging.ERROR)
    TrainComplete().run()
