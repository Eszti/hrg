from collections import defaultdict

from source.common.report import find_best_in_column, make_markdown_table
from source.common.scores.scoring_metrics_for_system import ScoringMetricsForSystem


class SystemScorer:
    def __init__(self, sentence_scorers):
        self.sentence_scorers_per_model = sentence_scorers
        self.sys_scores_per_model = defaultdict(dict)
        self.__get_scores()
        self.report = ""
        self.__get_report()

    def log_results(self, logger):
        for grammar_dir, scores_for_grammar in self.sys_scores_per_model.items():
            logger.log(f"Grammar dir: {grammar_dir}\n")
            for model_name, scores in scores_for_grammar.items():
                logger.log(f"{model_name}:\n{scores.to_str()}\n")

    def save_report(self, fn):
        with open(fn, "w") as f:
            f.writelines(self.report)

    def __get_scores(self):
        for (
            grammar_dir,
            sen_scorers_for_grammar,
        ) in self.sentence_scorers_per_model.items():
            for model_name, sen_scorers in sen_scorers_for_grammar.items():
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
                self.sys_scores_per_model[grammar_dir][model_name] = (
                    ScoringMetricsForSystem(metrics)
                )

    def __get_report(self):
        self.report = "# Evaluation\n"
        last_model_name = ""
        for grammar_dir, scores_for_grammar in self.sys_scores_per_model.items():
            table = []
            self.report += f"## {grammar_dir}\n"
            for model_name_with_k, scores in scores_for_grammar.items():
                current_model_name = model_name_with_k.split("_")[0]
                if last_model_name != current_model_name:
                    if len(table) > 0:
                        self.__add_table_to_report(table)
                    self.report += f"### {current_model_name}\n"
                    table = [
                        [
                            "model_name",
                            "predicted extractions",
                            "gold extractions",
                            "matches",
                            "exact matches",
                            "prec",
                            "rec",
                            "F1",
                        ]
                    ]
                table.append([model_name_with_k] + scores.get_values_for_report())
                last_model_name = current_model_name
            self.__add_table_to_report(table)

    def __add_table_to_report(self, table):
        bold = find_best_in_column(table, ["prec", "rec", "F1"])
        self.report += make_markdown_table(table, bold)
        self.report += "\n"
