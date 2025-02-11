from source.common.exceptions import ScoreDisorderException

from source.common.triplet.triplet import Triplet
from source.common.triplet.triplets_for_sen import TripletsForSen


class PostProcessedDerivationList:
    def __init__(self, derivation_list):
        self.derivation_list = derivation_list

    def save_derivations_as_graph_file(self, fn):
        with open(fn, "w") as f:
            for derivation in self.derivation_list:
                f.write(f"{derivation.print_shifted()};{derivation.score:g}\n")

    def get_triplets_for_sen(self, sen_id, sen_text):
        triplets = []
        for i, derivation in enumerate(self.derivation_list):
            triplets.append(
                Triplet.from_processed_triplet(
                    processed_triplet=derivation.processed_triplet,
                    score=derivation.score,
                    k=i + 1,
                )
            )
        return TripletsForSen(triplets, sen_id, sen_text)

    def check_score_disorder(self):
        for i in range(len(self.derivation_list) - 1):
            score_a = self.derivation_list[i].score
            score_b = self.derivation_list[i + 1].score
            if score_a < score_b:
                raise ScoreDisorderException(i, score_a, score_b)

    def log_all_derivations(self, logger):
        logger.log("=====================\n")
        for i, derivation in enumerate(self.derivation_list):
            logger.log(
                f"Processed score: {derivation.score:g} ({derivation.score_name})\n"
            )
            derivation.full_log(logger, i + 1)
            logger.log("=====================\n")
