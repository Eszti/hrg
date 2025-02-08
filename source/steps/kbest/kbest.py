import json
import os.path

from common.script.logger import Logger
from common.oie.triplet import Triplet
from source.common.conll import ConllSen
from source.common.script.loop_on_sen_dirs import LoopOnSenDirs
from common.bolinas.cky_chart import CkyChart


class KbestModel:
    def __init__(self):
        self.max_filter = False
        self.kbest = True
        self.k = 10
        self.subdir = None

    def get_derivation_per_model(
        self, derivation_list, gold_triplets, pos_tags, top_order, arg_perm=False
    ):
        if self.kbest:
            return {self.subdir: derivation_list.get_k_best_unique_derivation(self.k)}
        return derivation_list.get_best_matching_derivations(
            gold_triplets, pos_tags, top_order, arg_perm
        )


class BasicModel(KbestModel):
    def __init__(self):
        super().__init__()
        self.subdir = "basic"


class MaxModel(KbestModel):
    def __init__(self):
        super().__init__()
        self.subdir = "max"
        self.max_filter = True


class PRModel(KbestModel):
    def __init__(self):
        super().__init__()
        self.subdir = "pr_best"
        self.kbest = False


class KBest(LoopOnSenDirs):

    def __init__(self, config=None):
        super().__init__(
            description="Script to search k best derivations in parsed charts.",
            config=config,
        )
        self.logprob = True
        self.model_name_to_class = {
            "basic": BasicModel(),
            "max": MaxModel(),
            "pr_best": PRModel(),
        }

    def _do_for_sen(self, sen_idx, preproc_sen_dir):
        sen_dir = f"{self.out_dir}/{str(sen_idx)}"
        chart_file = f"{sen_dir}/parse/sen{sen_idx}_chart.pickle"
        if not os.path.exists(chart_file):
            print("Chart file path does not exist.")
            return

        cky_chart = CkyChart.from_pickle(chart_file)

        if cky_chart.no_derivation():
            print("No derivation found")
            return

        gold_triplets = KBest.__get_gold_triplets(preproc_sen_dir)
        top_order = json.load(open(f"{preproc_sen_dir}/pos_edge_graph_top_order.json"))
        pos_tags = ConllSen(preproc_sen_dir).pos_tags()

        kbest_dir = self._get_subdir("kbest", parent_dir=sen_dir)

        for model_name in sorted(self.config["models"]):
            model = self.model_name_to_class[model_name]
            out_dir = self._get_subdir(model.subdir, parent_dir=kbest_dir)

            sen_logger = Logger(f"{out_dir}/sen{sen_idx}.log")
            sen_logger.log(f"Processing {model_name}", to_stdout=True)

            filtered_chart = cky_chart
            sen_logger.log(f"\n{filtered_chart.log_length()}")
            if model.max_filter:
                sen_logger.log("Apply max size filter")
                filtered_chart = cky_chart.chart_with_only_max_size()
                sen_logger.log(filtered_chart.log_length())

            derivation_list = filtered_chart.search_derivations(
                "START", logger=sen_logger
            )
            derivations_per_model = model.get_derivation_per_model(
                derivation_list, gold_triplets, pos_tags, top_order
            )

            for submodel_name, derivations in derivations_per_model.items():
                derivations.check_score_disorder()
                sen_logger.log(f"Log derivations for {submodel_name}\n")
                derivations.log_all_derivations(sen_logger)
                derivations.save_derivations_as_graph_file(
                    f"{out_dir}/sen{sen_idx}_{submodel_name}_matches.graph"
                )
                derivations.save_triplets(
                    f"{out_dir}/sen{sen_idx}_{submodel_name}_triplets.txt"
                )

    @staticmethod
    def __get_gold_triplets(preproc_dir):
        gold_triplets = []
        files = [fn for fn in os.listdir(preproc_dir) if fn.endswith("_triplet.txt")]
        for fn in files:
            with open(f"{preproc_dir}/{fn}") as f:
                gold_triplets.append(Triplet(json.load(f)))
        return gold_triplets


if __name__ == "__main__":
    KBest().run()
