import json
import os.path

from source.common.bolinas.cky_chart import CkyChart
from source.common.conll import ConllSen
from source.common.script.logger import Logger
from source.common.script.loop_on_sen_dirs import LoopOnSenDirs
from source.common.triplet.triplets_for_sen import TripletsForSen


class KBestStep:

    def __init__(self, logger, models, gold_triplets_fn, pos_tag_resolution, arg_perm):
        self.logger = logger
        self.models = models
        self.gold_triplets_fn = gold_triplets_fn
        self.pos_tag_resolution = pos_tag_resolution
        self.arg_perm = arg_perm

        self.model_name_to_class = {
            "basic": BasicModel(),
            "max": MaxModel(),
            "pr_best": PRModel(),
        }

        self.no_derivation_found = []
        self.successful_derivation = 0
        self.all_sens = 0

    def search_sen(self, cky_chart: CkyChart, sen_idx, in_dir, out_dir):
        self.all_sens += 1
        if cky_chart.no_derivation():
            self.logger.log("No derivation found", to_stdout=True)
            self.no_derivation_found += 1
            return

        self.successful_derivation += 1
        gold_triplets_for_sen = TripletsForSen.from_json(
            f"{in_dir}/{self.gold_triplets_fn}"
        )
        gold_triplets = gold_triplets_for_sen.triplets
        conll_sen = ConllSen(in_dir)
        sen_text = conll_sen.sen_text()

        for model_name in sorted(self.models):
            model = self.model_name_to_class[model_name]

            sen_logger = Logger(f"{out_dir}/sen{sen_idx}_{model_name}.log")
            sen_logger.log(f"Processing {model_name}", to_stdout=True)
            sen_logger.log(f"\nGold triplets:\n{gold_triplets_for_sen.to_short_json()}")

            filtered_chart = cky_chart
            sen_logger.log(f"{filtered_chart.log_length()}")
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
                sen_id=sen_idx,
                sen_text=sen_text,
                pos_tag_resolution=self.pos_tag_resolution,
                arg_perm=self.arg_perm,
            )

            for submodel_name, derivations in derivations_per_model.items():
                derivations.check_score_disorder()
                sen_logger.log(f"Log derivations for {submodel_name}\n")
                derivations.log_all_derivations(sen_logger)
                triplets_for_sen = derivations.triplets_for_sen
                triplets_for_sen.save_summary(
                    f"{out_dir}/sen{sen_idx}_{submodel_name}_triplets_summary.txt"
                )
                triplets_for_sen.to_json(
                    f"{out_dir}/sen{sen_idx}_{submodel_name}_triplets.json"
                )

    def log_kbest_step(self):
        self.logger.log(
            f"\nNumber of no derivation found: {len(self.no_derivation_found)}\n"
            f"{json.dumps(self.no_derivation_found)}"
            f"\nNumber of successful derivation: {self.successful_derivation}"
            f"\nNumber of all sentences: {self.all_sens}",
            to_stdout=True,
        )


class KbestModel:
    def __init__(self):
        self.max_filter = False
        self.kbest = True
        self.k = 10
        self.subdir = None

    def get_derivation_per_model(
        self,
        derivation_list,
        gold_triplets,
        sen_id,
        sen_text,
        pos_tag_resolution,
        arg_perm,
    ):
        if self.kbest:
            return {
                self.subdir: derivation_list.get_k_best_unique_derivation(
                    sen_id=sen_id,
                    sen_text=sen_text,
                    k=self.k,
                    pos_tag_resolution=pos_tag_resolution,
                )
            }
        return derivation_list.get_best_matching_derivations(
            sen_id=sen_id,
            sen_text=sen_text,
            gold_triplets=gold_triplets,
            pos_tag_resolution=pos_tag_resolution,
            arg_perm=arg_perm,
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

    def __init__(self, description=None, script_name=None, config=None):
        if description is None:
            description = "Script to search k best derivations in parsed charts."
        if script_name is None:
            script_name = "kbest"
        super().__init__(
            description=description,
            script_name=script_name,
            config=config,
        )
        self.gold_triplets_fn = "gold_triplets.json"
        self.pos_tag_resolution = False
        self.pred_resolution_from_rule = False
        self.arg_perm = True

        self.no_chart = []
        self.chart_load_failed = []
        self.all_sens = 0

    def _before_loop(self):
        self.kbest_step = KBestStep(
            logger=self.logger,
            models=self.config["models"],
            gold_triplets_fn=self.gold_triplets_fn,
            pos_tag_resolution=self.pos_tag_resolution,
            arg_perm=self.arg_perm,
        )

    def _do_for_sen(self, sen_idx, preproc_sen_dir):
        sen_dir = f"{self.out_dir}/{str(sen_idx)}"
        kbest_dir = self._get_subdir("kbest", parent_dir=sen_dir)

        chart_file = f"{sen_dir}/parse/sen{sen_idx}_chart.pickle"
        self.logger.log(f"Parsing sen {sen_idx}")
        self.all_sens += 1

        if not os.path.exists(chart_file):
            self.logger.log("Chart file path does not exist.", to_stdout=True)
            self.no_chart.append(sen_idx)
            return

        try:
            cky_chart = CkyChart.from_pickle(chart_file)
        except KeyError:
            self.chart_load_failed.append(sen_idx)
            return

        self.kbest_step.search_sen(
            cky_chart=cky_chart,
            sen_idx=sen_idx,
            in_dir=preproc_sen_dir,
            out_dir=kbest_dir,
        )

    def _after_loop(self):
        self.kbest_step.log_kbest_step()
        self.logger.log(
            f"\nNumber of no chart: {len(self.no_chart)}\n"
            f"{json.dumps(self.no_chart)}"
            f"\nNumber of chart load failed: {len(self.chart_load_failed)}\n"
            f"{json.dumps(self.chart_load_failed)}"
            f"\nNumber of all sentences: {self.all_sens}",
            to_stdout=True,
        )
        super()._after_loop()


if __name__ == "__main__":
    KBest().run()
