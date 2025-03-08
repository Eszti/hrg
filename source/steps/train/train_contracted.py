import json
import logging
import re

import networkx as nx
from tuw_nlp.graph.graph import Graph

from source.common.rules.hrg_for_triplet import HRGForTriplet
from source.common.rules.hrg_rule import HRGRule
from source.steps.train.train import Train


class TrainContracted(Train):
    def __init__(self, config=None):
        super().__init__(
            description="Script to create hrg rules on preprocessed arg contracted train data.",
            script_name="train_contracted",
            config=config,
        )
        self.pos_tag_resolution = True
        self.pred_resolution_from_rule = True
        self.gold_triplets_fn = "gold_contracted_triplets.json"
        self.unconnected_pred = []
        self.out_edge_from_pred_subgraph = []

    def _get_hrg_for_triplet(
        self, triplet_graph_str, triplet_graph, triplet_id, triplet, triplet_logger
    ):
        pred_subtree = self.__get_pred_subtree(triplet, triplet_id, triplet_graph)
        if pred_subtree is None:
            return HRGForTriplet([], triplet_id)
        pred_rule = self.__get_hrg_rules_for_predicate(pred_subtree, triplet_id)
        initial_rule = self.__get_initial_rule(pred_subtree, triplet_graph, triplet_id)
        hrg_for_triplet = HRGForTriplet([initial_rule, pred_rule], triplet_id)
        return hrg_for_triplet

    @staticmethod
    def __get_initial_rule(pred_subtree, triplet_graph, triplet_id):
        roots = [n for n in list(pred_subtree.nodes) if pred_subtree.in_degree(n) == 0]
        assert len(roots) == 1
        pred_root = roots[0]
        nodes_to_remove = [n for n in list(pred_subtree.nodes) if n not in roots]
        triplet_graph.G.remove_nodes_from(nodes_to_remove)
        initial_rule_rhs = triplet_graph.to_bolinas(
            keep_node_ids=True, add_n_prefix=False
        )
        initial_rule_rhs = initial_rule_rhs.replace(
            f"({pred_root}.", f"({pred_root}. :P$"
        )
        initial_rule_rhs = initial_rule_rhs.replace(
            f" {pred_root}.", f" ({pred_root}. :P$)"
        )
        initial_rule_rhs = re.sub(r"n[0-9]*\.", ".", initial_rule_rhs)
        initial_rule = HRGRule(
            lhs="S",
            rhs_string=initial_rule_rhs,
            triplet_id=triplet_id,
        )
        return initial_rule

    @staticmethod
    def __get_hrg_rules_for_predicate(pred_subtree, triplet_id):
        subtree_string = Graph.from_networkx(pred_subtree).to_bolinas(
            add_n_prefix=False, keep_node_ids=False
        )
        pred_rule = HRGRule(
            lhs="P",
            rhs_string=subtree_string,
            triplet_id=triplet_id,
        )
        return pred_rule

    def __get_pred_subtree(self, triplet, triplet_id, triplet_graph):
        predicates = []
        for p in triplet.predicate():
            predicates += [f"n{p}", f"n{1000+p}"]
        pred_subtree = triplet_graph.G.subgraph(predicates)
        if not nx.is_weakly_connected(pred_subtree):
            self.unconnected_pred.append(triplet_id)
            return None
        for n in predicates:
            if (
                triplet_graph.G.out_degree(n) > pred_subtree.out_degree(n)
                and pred_subtree.in_degree(n) != 0
            ):
                self.out_edge_from_pred_subgraph.append(triplet_id)
                return None
        return pred_subtree

    def _after_loop(self):
        self.logger.log(
            f"\nNumber of unconnected predicate: {len(self.unconnected_pred)}\n"
            f"{json.dumps(self.unconnected_pred)}"
            f"\nNumber of out edge from predicate subgraph: {len(self.out_edge_from_pred_subgraph)}\n"
            f"{json.dumps(self.out_edge_from_pred_subgraph)}",
            to_stdout=True,
        )
        super()._after_loop()


if __name__ == "__main__":
    logging.getLogger("penman").setLevel(logging.ERROR)
    TrainContracted().run()
