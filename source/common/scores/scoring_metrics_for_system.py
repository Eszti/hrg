from source.common.triplet.triplet_matcher import TripletMatcher


class ScoringMetricsForSystem:
    def __init__(self, scores_dict):
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

    def get_values_for_report(self):
        ret = [
            self.precision[0][1],
            self.recall[0][1],
            self.precision_of_matches[0][1],
            self.exact_matches_precision[0][0],
        ]
        prec, rec = self.precision[1], self.recall[1]
        f1_score = round(TripletMatcher.f1(prec, rec), 4)
        prec, rec = round(prec, 4), round(rec, 4)
        ret.append(prec)
        ret.append(rec)
        ret.append(f1_score)
        return ret
