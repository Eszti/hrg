from abc import abstractmethod

from source.common.bolinas.parser import Parser
from source.common.script.logger import Logger
from source.common.script.loop_on_sen_dirs import LoopOnSenDirs
from source.steps.kbest.kbest import KBestStep
from source.steps.parse.parse import ParseStep


class Bolinas(LoopOnSenDirs):

    def __init__(self, description=None, script_name=None, config=None):
        if description is None:
            description = "Script to parse graphs and search for k best derivations."
        if script_name is None:
            script_name = "bolinas"
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
        self._load_grammar()
        parser = Parser(self.grammar, max_steps=self.config.get("max_steps"))
        self.parse_step = ParseStep(parser, self.logger)
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

        self.logger.log(f"Processing sen {sen_idx}")

        for triplet_id, graph_file in self._get_triplet_input(sen_idx, preproc_sen_dir):
            sen_logger = Logger(f"{kbest_dir}/sen{str(triplet_id)}_parse.log")
            cky_chart = self.parse_step.parse_sen(
                triplet_id=triplet_id,
                graph_file=graph_file,
                sen_logger=sen_logger,
            )
            if cky_chart is not None:
                self.kbest_step.search_sen(
                    cky_chart=cky_chart,
                    sen_idx=sen_idx,
                    in_dir=preproc_sen_dir,
                    out_dir=kbest_dir,
                )

    @abstractmethod
    def _get_triplet_input(self, sen_idx, sen_dir):
        raise NotImplemented

    def _after_loop(self):
        self.logger.log(
            f"\n====================\nParse log\n====================", to_stdout=True
        )
        self.parse_step.log_parse_step()
        self.logger.log(
            f"\n====================\nKBest log\n====================", to_stdout=True
        )
        self.kbest_step.log_kbest_step()
        self.logger.log(
            f"\n====================\nBolinas log\n====================\n",
            to_stdout=True,
        )
        super()._after_loop()
