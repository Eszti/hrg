class ScoringMetricsForSen:
    def __init__(self, matches_summary, exact_matches_summary):
        self.precision = matches_summary["precision"]
        self.recall = matches_summary["recall"]
        self.precision_of_matches = matches_summary["precision_of_matches"]
        self.recall_of_matches = matches_summary["recall_of_matches"]
        self.precision_of_exact_matches = exact_matches_summary["exact_match_precision"]
        self.recall_of_exact_matches = exact_matches_summary["exact_match_recall"]

    def to_str(self):
        json_dict = self.__dict__
        ret = ""
        for key in json_dict:
            ret += key + ": " + str(json_dict[key]) + "\n"
        return ret
