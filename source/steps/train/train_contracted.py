import logging

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

    def _get_hrg_for_triplet(
        self, triplet_graph_str, triplet_id, triplet, triplet_logger
    ):
        hrg_rule = HRGRule(
            lhs="S",
            rhs_string=triplet_graph_str,
            triplet=triplet,
            triplet_id=triplet_id,
            predicate_info=True,
        )
        triplet_logger.log(hrg_rule.log_hrg_rule())
        return HRGForTriplet([hrg_rule], triplet_id)


if __name__ == "__main__":
    logging.getLogger("penman").setLevel(logging.ERROR)
    TrainContracted().run()
