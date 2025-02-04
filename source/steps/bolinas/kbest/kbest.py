import json
import os.path

from common.script.logger import Logger
from source.common.conll import ConllSen
from source.common.script.loop_on_sen_dirs import LoopOnSenDirs
from source.steps.bolinas.kbest.filter.pr_filter import filter_for_pr
from steps.bolinas.common.cky_chart import CkyChart


def get_gold_labels(preproc_dir):
    gold_labels = []
    files = [fn for fn in os.listdir(preproc_dir) if fn.endswith("_triplet.txt")]
    for fn in files:
        with open(f"{preproc_dir}/{fn}") as f:
            gold_labels.append(json.load(f))
    return gold_labels


class KBest(LoopOnSenDirs):

    def __init__(self, config=None):
        super().__init__(description="Script to search k best derivations in parsed charts.", config=config)
        self.logprob = True
        self.score_disorder_collector = {}

    def _before_loop(self):
        pass

    def _do_for_sen(self, sen_idx, sen_dir):
        bolinas_dir = f"{self.out_dir}/{str(sen_idx)}/bolinas"
        chart_file = f"{bolinas_dir}/sen{sen_idx}_chart.pickle"
        if not os.path.exists(chart_file):
            return

        cky_chart = CkyChart.from_pickle(chart_file)

        if cky_chart.no_derivation():
            print("No derivation found")
            return

        gold_labels = get_gold_labels(sen_dir)
        top_order = json.load(open(
            f"{sen_dir}/pos_edge_graph_top_order.json"
        ))
        pos_tags = ConllSen(sen_dir).pos_tags()

        for name, c in sorted(self.config["filters"].items()):
            if c.get("ignore", False):
                continue
            print(f"Processing {name}")
            out_dir = self._get_subdir(name, parent_dir=bolinas_dir)
            sen_logger = Logger(f"{out_dir}/sen{sen_idx}.log")

            filtered_chart = cky_chart
            sen_logger.log(filtered_chart.log_length())
            if "chart_filter" in c and c["chart_filter"] == "max":
                sen_logger.log("Apply max size filter")
                filtered_chart = cky_chart.chart_with_only_max_size()
                sen_logger.log(filtered_chart.log_length())

            derivation_list = filtered_chart.search_derivations("START", logger=sen_logger)

            assert ("k" in c and "pr_metric" not in c) or ("k" not in c and "pr_metric" in c)

            if "k" in c:
                k_best_unique_derivations = derivation_list.get_k_best_unique_derivation(c["k"])
            elif "pr_metric" in c:
                metric = c["pr_metric"]
                assert metric in ["prec", "rec", "f1"]
                k_best_unique_derivations, labels_with_arg_idx = filter_for_pr(
                    derivations,
                    gold_labels,
                    metric,
                    pos_tags,
                    top_order,
                    self.config["arg_permutation"],
                )
                if len(k_best_unique_derivations) == 0:
                    sen_log_lines.append("No matches\n")
            else:
                print("Neither 'k' nor 'pr_metric' is set")
                continue

            self.score_disorder_collector[sen_idx] = k_best_unique_derivations.check_score_disorder(sen_logger)

            k_best_unique_derivations.log_all_derivations(sen_logger)
            k_best_unique_derivations.save_derivations_as_graph_file(f"{out_dir}/sen{sen_idx}_matches.graph")
            k_best_unique_derivations.save_predicted_labels(f"{out_dir}/sen{sen_idx}_predicted_labels.txt")

    def _after_loop(self):
        num_sem = len(self.score_disorder_collector.keys())
        self.logger.log(f"\nNumber of sentences: {num_sem}")
        sum_score_disorder = sum(self.score_disorder_collector.values())
        self.logger.log(f"Sum of score disorders: {sum_score_disorder}")
        self.logger.log(f"Average score disorders: {round(sum_score_disorder / float(num_sem), 2)}")
        super()._after_loop()


if __name__ == "__main__":
    KBest().run()
