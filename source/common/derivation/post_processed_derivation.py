from source.common.derivation.processed_derivation import ProcessedDerivation
from source.common.triplet.processed_triplet import ProcessedTriplet


class PostProcessedDerivation(ProcessedDerivation):

    def __init__(
        self, raw_derivation, pos_tags, top_order, score=None, score_name=None
    ):
        super().__init__(raw_derivation, score, score_name)
        self.processed_triplet = ProcessedTriplet(
            self.derived_labels,
            pos_tags=pos_tags,
            top_order=top_order,
            label_to_nodes=False,
        )

    def full_log(self, logger, k):
        super().full_log(logger, k)
        self.__log_triplet(logger)

    def __log_triplet(self, logger):
        logger.log(
            f"Derived labels:\n{self.raw_triplet.to_short_json()} - # nodes: {self.raw_triplet.len()}"
        )
        logger.log(
            f"Post-processed labels:\n{self.processed_triplet.to_short_json()} - # nodes: {self.processed_triplet.len()}\n"
        )
