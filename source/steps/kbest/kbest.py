import json
import os.path

from source.common.bolinas.cky_chart import CkyChart
from source.common.conll import ConllSen
from source.common.script.logger import Logger
from source.common.script.loop_on_sen_dirs import LoopOnSenDirs
from source.common.triplet.triplets_for_sen import TripletsForSen


class KbestModel:
    def __init__(self):
        self.max_filter = False
        self.kbest = True
        self.k = 10
        self.subdir = None

    def get_derivation_per_model(
        self, derivation_list, gold_triplets, pos_tags, top_order, sen_id, sen_text
    ):
        if self.kbest:
            return {
                self.subdir: derivation_list.get_k_best_unique_derivation(
                    sen_id=sen_id,
                    sen_text=sen_text,
                    k=self.k,
                    pos_tags=pos_tags,
                    top_order=top_order,
                )
            }
        return derivation_list.get_best_matching_derivations(
            sen_id=sen_id,
            sen_text=sen_text,
            gold_triplets=gold_triplets,
            top_order=top_order,
            pos_tags=pos_tags,
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
            script_name="kbest",
            config=config,
        )
        self.logprob = True
        self.model_name_to_class = {
            "basic": BasicModel(),
            "max": MaxModel(),
            "pr_best": PRModel(),
        }
        self.no_chart = 0
        self.no_derivation_found = 0
        self.successful_derivation = 0
        self.all_sens = 0

    def _do_for_sen(self, sen_idx, preproc_sen_dir):
        sen_dir = f"{self.out_dir}/{str(sen_idx)}"
        chart_file = f"{sen_dir}/parse/sen{sen_idx}_chart.pickle"
        self.logger.log(f"Parsing sen {sen_idx}")
        self.all_sens += 1

        if not os.path.exists(chart_file):
            self.logger.log("Chart file path does not exist.", to_stdout=True)
            self.no_chart += 1
            return

        cky_chart = CkyChart.from_pickle(chart_file)

        if cky_chart.no_derivation():
            self.logger.log("No derivation found", to_stdout=True)
            self.no_derivation_found += 1
            return

        self.successful_derivation += 1
        gold_triplets = TripletsForSen.from_json(
            f"{preproc_sen_dir}/gold_triplets.json"
        ).triplets
        top_order = json.load(open(f"{preproc_sen_dir}/graph_top_order.json"))
        conll_sen = ConllSen(preproc_sen_dir)
        pos_tags = conll_sen.pos_tags()
        sen_text = conll_sen.sen_text()

        kbest_dir = self._get_subdir("kbest", parent_dir=sen_dir)

        for model_name in sorted(self.config["models"]):
            model = self.model_name_to_class[model_name]

            sen_logger = Logger(f"{kbest_dir}/sen{sen_idx}_{model_name}.log")
            sen_logger.log(f"Processing {model_name}", to_stdout=True)

            filtered_chart = cky_chart
            sen_logger.log(f"\n{filtered_chart.log_length()}")
            if model.max_filter:
                sen_logger.log("Apply max size filter")
                filtered_chart = cky_chart.chart_with_only_max_size()
                sen_logger.log(filtered_chart.log_length())

            derivation_list = filtered_chart.search_derivations(
                "START", sen_logger=sen_logger, global_logger=self.logger
            )
            derivations_per_model = model.get_derivation_per_model(
                derivation_list=derivation_list,
                gold_triplets=gold_triplets,
                pos_tags=pos_tags,
                top_order=top_order,
                sen_id=sen_idx,
                sen_text=sen_text,
            )

            for submodel_name, derivations in derivations_per_model.items():
                derivations.check_score_disorder()
                sen_logger.log(f"Log derivations for {submodel_name}\n")
                derivations.log_all_derivations(sen_logger)
                triplets_for_sen = derivations.triplets_for_sen
                triplets_for_sen.save_summary(
                    f"{kbest_dir}/sen{sen_idx}_{submodel_name}_triplets_summary.txt"
                )
                triplets_for_sen.to_json(
                    f"{kbest_dir}/sen{sen_idx}_{submodel_name}_triplets.json"
                )

    def _after_loop(self):
        self.logger.log(
            f"\nNumber of no chart: {self.no_chart}"
            f"\nNumber of no derivation found: {self.no_derivation_found}"
            f"\nNumber of successful derivation: {self.successful_derivation}"
            f"\nNumber of all sentences: {self.all_sens}",
            to_stdout=True,
        )
        super()._after_loop()


if __name__ == "__main__":
    KBest().run()
