import logging

from source.steps.train.train import Train


class TrainContracted(Train):
    def __init__(self, config=None):
        super().__init__(
            description="Script to create hrg rules on preprocessed arg contracted train data.",
            script_name="train_contracted",
            config=config,
        )
        self.pos_tag_resolution = True

    def _get_rules(self, triplet_graph, triplet):
        return {
            "initial_rule": f"S -> {triplet_graph.to_bolinas(keep_node_ids=False, add_n_prefix=False)};",
            "rules": [],
        }


if __name__ == "__main__":
    logging.getLogger("penman").setLevel(logging.ERROR)
    TrainContracted().run()
