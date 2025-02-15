from source.common.scores.scoring_metrics_for_system import ScoringMetricsForSystem


class SystemScorer:
    def __init__(self, sentence_scorers):
        self.sentence_scorers_per_model = sentence_scorers
        self.sys_scores_per_model = {}
        self.__get_scores()

    def __get_scores(self):
        for model_name, sen_scorers in self.sentence_scorers_per_model.items():
            prec_num, prec_denominator = 0, 0
            rec_num, rec_denominator = 0, 0
            exact_prec_num, exact_prec_denominator = 0, 0
            exact_rec_num, exact_rec_denominator = 0, 0
            tot_prec_of_matches, tot_rec_of_matches = 0, 0
            matches_len = 0

            for sen_scorer in sen_scorers:
                matches_len += len(sen_scorer.matches)
                s = sen_scorer.scoring_metrics
                prec_num += s.precision[0]
                prec_denominator += s.precision[1]
                rec_num += s.recall[0]
                rec_denominator += s.recall[1]
                exact_prec_num += s.precision_of_exact_matches[0]
                exact_prec_denominator += s.precision_of_exact_matches[1]
                exact_rec_num += s.recall_of_exact_matches[0]
                exact_rec_denominator += s.recall_of_exact_matches[1]
                tot_prec_of_matches += sum(s.precision_of_matches)
                tot_rec_of_matches += sum(s.recall_of_matches)

            metrics = {
                "precision": (
                    [prec_num, prec_denominator],
                    prec_num / prec_denominator,
                ),
                "recall": ([rec_num, rec_denominator], rec_num / rec_denominator),
                "matches_len": matches_len,
                "precision_of_matches": (
                    [tot_prec_of_matches, matches_len],
                    tot_prec_of_matches / matches_len,
                ),
                "recall_of_matches": (
                    [tot_rec_of_matches, matches_len],
                    tot_rec_of_matches / matches_len,
                ),
                "exact_matches_precision": (
                    [exact_prec_num, exact_prec_denominator],
                    exact_prec_num / exact_prec_denominator,
                ),
                "exact_matches_recall": (
                    [exact_rec_num, exact_rec_denominator],
                    exact_rec_num / exact_rec_denominator,
                ),
            }
            self.sys_scores_per_model[model_name] = ScoringMetricsForSystem(metrics)

    def log_results(self, logger):
        for model_name, scores in self.sys_scores_per_model.items():
            logger.log(f"{model_name}:\n{scores.to_str()}\n")
