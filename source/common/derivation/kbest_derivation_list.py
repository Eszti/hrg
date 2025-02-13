from source.common.exceptions import ScoreDisorderException

from source.common.triplet.triplets_for_sen import TripletsForSen


class KbestDerivationList:
    def __init__(self, derivation_list, sen_id, sen_text):
        self.derivation_list = derivation_list
        self.triplets_for_sen = TripletsForSen(
            [d.processed_triplet for d in self.derivation_list], sen_id, sen_text
        )

    def check_score_disorder(self):
        for i in range(len(self.derivation_list) - 1):
            score_a = self.derivation_list[i].score
            score_b = self.derivation_list[i + 1].score
            if score_a < score_b:
                raise ScoreDisorderException(i, score_a, score_b)

    def log_all_derivations(self, logger):
        logger.log("=====================\n")
        for i, derivation in enumerate(self.derivation_list):
            derivation.full_log(logger, i + 1)
            logger.log("=====================\n")
