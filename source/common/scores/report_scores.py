from source.common.scores.scoring_metrics_for_system import ScoringMetricsForSystem
from source.common.triplet.triplet_matcher import TripletMatcher


class ReportScores:

    header_names = [
        "model_name",
        "predicted extractions",
        "gold extractions",
        "matches",
        "exact matches",
        "prec of matches",
        "rec of matches",
        "prec (only exact)",
        "rec (only exact)",
        "f1 (only exact)",
        "prec",
        "rec",
        "F1",
    ]

    def __init__(self, scores_for_sen: ScoringMetricsForSystem):
        self.scores_dict = {
            "model_name": scores_for_sen.model_name,
            "predicted extractions": scores_for_sen.precision[0][1],
            "gold extractions": scores_for_sen.recall[0][1],
            "matches": scores_for_sen.precision_of_matches[0][1],
            "exact matches": scores_for_sen.exact_matches_precision[0][0],
            "prec of matches": round(scores_for_sen.precision_of_matches[1], 4),
            "rec of matches": round(scores_for_sen.recall_of_matches[1], 4),
            "prec (only exact)": round(scores_for_sen.exact_matches_precision[1], 4),
            "rec (only exact)": round(scores_for_sen.exact_matches_recall[1], 4),
            "f1 (only exact)": round(
                TripletMatcher.f1(
                    scores_for_sen.exact_matches_precision[1],
                    scores_for_sen.exact_matches_recall[1],
                ),
                4,
            ),
            "prec": round(scores_for_sen.precision[1], 4),
            "rec": round(scores_for_sen.recall[1], 4),
            "F1": round(
                TripletMatcher.f1(
                    scores_for_sen.precision[1], scores_for_sen.recall[1]
                ),
                4,
            ),
        }

    def get_scores(self):
        return [self.scores_dict[h] for h in ReportScores.header_names]
