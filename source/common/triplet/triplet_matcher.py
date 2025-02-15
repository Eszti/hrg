class TripletMatcher:
    def __init__(self, gold, predicted):
        self.gold = gold
        self.predicted = predicted
        self.exact_match = self.__exact_match()
        self.match = self.__match()
        self.scores = self.__get_scores()

    def __get_scores(self):
        metrics = {"prec": 0, "rec": 0, "f1": 0}
        if self.match:
            g_and_p = self.__calc_p_and_g()
            p_len = self.predicted.len()
            g_len = self.gold.len()
            metrics["prec"] = g_and_p / float(p_len)
            metrics["rec"] = g_and_p / float(g_len)
            prec = g_and_p / float(p_len)
            rec = g_and_p / float(g_len)
            metrics["f1"] = self.__f1(prec, rec)
        return metrics

    def __calc_p_and_g(self):
        ret = 0
        for label, gold_nodes in self.gold.label_to_nodes.items():
            if label.startswith("P") or label.startswith("A"):
                predicted_nodes = self.predicted.label_to_nodes.get(label, [])
                ret += len(set(gold_nodes) & set(predicted_nodes))
        return ret

    @staticmethod
    def __f1(prec, rec):
        try:
            return 2 * prec * rec / (prec + rec)
        except ZeroDivisionError:
            return 0

    def __match(self):
        p_match = self.gold.predicate() == self.predicted.predicate()
        a0_match = self.gold.a0() == self.predicted.a0()
        return p_match and a0_match

    def __exact_match(self):
        return self.gold.label_to_nodes == self.predicted.label_to_nodes
