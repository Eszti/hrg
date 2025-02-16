class ScoringMetricsForSystem:
    def __init__(self, scores_dict, model_name):
        self.model_name = model_name
        self.matches_len = scores_dict["matches_len"]
        self.precision = scores_dict["precision"]
        self.recall = scores_dict["recall"]
        self.precision_of_matches = scores_dict["precision_of_matches"]
        self.recall_of_matches = scores_dict["recall_of_matches"]
        self.exact_matches_precision = scores_dict["exact_matches_precision"]
        self.exact_matches_recall = scores_dict["exact_matches_recall"]

    def to_str(self):
        json_dict = self.__dict__
        ret = ""
        for key in json_dict:
            ret += key + ": " + str(json_dict[key]) + "\n"
        return ret
