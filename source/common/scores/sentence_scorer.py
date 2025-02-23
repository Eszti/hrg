from source.common.scores.scoring_metrics_for_sen import ScoringMetricsForSen
from source.common.triplet.triplet_matcher import TripletMatcher


class SentenceScorer:
    def __init__(self, gold, pred, arg_perm=False, strict=True):
        self.gold = gold.triplets
        self.pred = pred.triplets
        self.strict = strict
        if arg_perm:
            self.__use_best_permutations()
        self.exact_match_scores = [[None for _ in self.pred] for __ in self.gold]
        self.scores = [[None for _ in self.pred] for __ in self.gold]
        self.exact_matches = []
        self.matches = []
        self.scoring_metrics = None
        self.__get_scores()

    def to_file(self, fn):
        with open(fn, "w") as f:
            f.write(f"Gold triplets:\n")
            for triplet in self.gold:
                f.write(f"{triplet.to_short_json()}\n")
            f.write(f"\nPred triplets:\n")
            for triplet in sorted(self.pred, key=lambda x: x.triplet_id):
                f.write(f"{triplet.to_short_json()}\n")
            exact_matches_str = "\n".join([str(m) for m in self.exact_matches])
            f.write(f"\nExact matches:\n{exact_matches_str}\n")
            matches_str = "\n".join([str(m) for m in self.matches])
            f.write(f"\nMatches:\n{matches_str}\n")
            f.write(f"\nScoring metrics:\n{self.scoring_metrics.to_str()}\n")

    def __use_best_permutations(self):
        new_pred = []
        for orig_p in self.pred:
            matches = []
            permutations = orig_p.get_all_permutations()
            for g in self.gold:
                for p in permutations:
                    matcher = TripletMatcher(g, p, self.strict)
                    if matcher.match:
                        matches.append((p, matcher.scores["f1"]))
            if len(matches) >= 1:
                matches = sorted(matches, key=lambda x: x[1], reverse=True)
                new_pred.append(matches[0][0])
            else:
                new_pred.append(orig_p)
        self.pred = new_pred

    def __get_scores(self):
        for i, gt in enumerate(self.gold):
            for j, pt in enumerate(self.pred):
                matcher = TripletMatcher(gt, pt, self.strict)
                self.exact_match_scores[i][j] = matcher.exact_match
                if matcher.exact_match:
                    self.exact_matches.append(
                        {"gold": gt.triplet_id, "pred": pt.triplet_id}
                    )
                self.scores[i][j] = matcher.scores
        matches_summary = self.__aggregate_scores_greedily()
        exact_match_summary = self.__aggregate_exact_matches()
        self.scoring_metrics = ScoringMetricsForSen(
            matches_summary, exact_match_summary
        )

    def __aggregate_scores_greedily(self):
        matches = []
        while True:
            max_s = 0
            gold, pred = None, None
            for i, gold_ss in enumerate(self.scores):
                if i in [m[0] for m in matches]:
                    continue
                for j, pred_s in enumerate(self.scores[i]):
                    if j in [m[1] for m in matches]:
                        continue
                    if pred_s and pred_s["f1"] > max_s:
                        max_s = pred_s["f1"]
                        gold = i
                        pred = j
            if max_s == 0:
                break
            matches.append([gold, pred])
            self.matches.append(
                {
                    "gold": self.gold[gold].triplet_id,
                    "pred": self.pred[pred].triplet_id,
                    "scores": self.scores[gold][pred],
                }
            )
        prec_scores = [self.scores[i][j]["prec"] for i, j in matches]
        total_prec = sum(prec_scores)
        rec_scores = [self.scores[i][j]["rec"] for i, j in matches]
        total_rec = sum(rec_scores)
        # Todo
        f1_scores = [self.scores[i][j]["f1"] for i, j in matches]
        total_f1 = sum(f1_scores)
        metrics = {
            "precision": [total_prec, len(self.scores[0])],
            "recall": [total_rec, len(self.scores)],
            "precision_of_matches": prec_scores,
            "recall_of_matches": rec_scores,
        }
        return metrics

    def __aggregate_exact_matches(self):
        recall = [
            sum([any(gold_matches) for gold_matches in self.exact_match_scores], 0),
            len(self.exact_match_scores),
        ]
        if len(self.exact_match_scores[0]) == 0:
            precision = [0, 0]
        else:
            precision = [
                sum(
                    [
                        any([g[i] for g in self.exact_match_scores])
                        for i in range(len(self.exact_match_scores[0]))
                    ],
                    0,
                ),
                len(self.exact_match_scores[0]),
            ]
        metrics = {"exact_match_precision": precision, "exact_match_recall": recall}
        return metrics
