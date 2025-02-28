import json
import logging

from source.common.bolinas.hgraph import Hgraph
from source.steps.train.train import Train


class TrainContracted(Train):
    def __init__(self, config=None):
        super().__init__(
            description="Script to create hrg rules on preprocessed arg contracted train data.",
            script_name="train_contracted",
            config=config,
        )
        self.pos_tag_resolution = True
        self.gold_triplets_fn = "gold_contracted_triplets.json"

    def _get_rules(self, triplet_graph_str, triplet, triplet_logger):
        rhs = Hgraph.from_string(triplet_graph_str)
        rhs_to_save = Hgraph.from_string(rhs.to_bolinas_str(nodeids=True))
        rhs_to_save.fill_node_order()
        predicates = "_".join(
            [str(rhs_to_save.node_to_order[f"n{p}"]) for p in triplet.predicate()]
        )
        initial_rule = f"S -> {rhs_to_save.to_string()};{predicates};"
        triplet_logger.log(
            f"Rule rhs with ids:\n{rhs_to_save.to_bolinas_str(nodeids=True)}"
            f"\nNodes to order:\n{json.dumps(rhs_to_save.node_to_order)}"
            f"\nPredicate nodes: {triplet.predicate()}"
            f"\nPredicate node orders: {predicates}\n"
            f"\nInitial rule:\n{initial_rule}\n"
        )
        return {
            "initial_rule": initial_rule,
            "rules": [],
        }


if __name__ == "__main__":
    logging.getLogger("penman").setLevel(logging.ERROR)
    TrainContracted().run()
