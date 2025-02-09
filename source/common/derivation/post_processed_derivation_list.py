from source.common.exceptions import ScoreDisorderException


class PostProcessedDerivationList:
    def __init__(self, derivation_list):
        self.derivation_list = derivation_list

    def __iter__(self):
        return (x for x in list.__iter__(self.derivation_list))

    def __getitem__(self, item):
        return self.derivation_list.__getitem__(item)

    def save_derivations_as_graph_file(self, fn):
        with open(fn, "w") as f:
            for derivation in self.derivation_list:
                f.write(f"{derivation.print_shifted()};{derivation.score:g}\n")

    def save_triplets(self, fn):
        with open(fn, "w") as f:
            for derivation in self.derivation_list:
                f.write(
                    f"{derivation.post_processed_triplet.to_json_str()};{derivation.score:g}\n"
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
            logger.log(
                f"Processed score: {derivation.score:g} ({derivation.score_name})\n"
            )
            derivation.full_log(logger, i + 1)
            logger.log("=====================\n")
