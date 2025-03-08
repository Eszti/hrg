import json
from abc import abstractmethod

from tuw_nlp.graph.graph import Graph

from source.common.bolinas.grammar import Grammar
from source.common.bolinas.parser import Parser
from source.common.bolinas.vo_rule import VoRule
from source.common.exceptions import (
    ParseTooLongException,
    CkyTooLongException,
    NotAllNodesCoveredException,
)
from source.common.rules.hrg_for_dataset import HRGForDataset
from source.common.script.logger import Logger
from source.common.script.loop_on_triplets import LoopOnTriplets
from source.common.triplet.triplet_matcher import TripletMatcher


class Train(LoopOnTriplets):
    def __init__(self, description, script_name, config=None):
        super().__init__(description, script_name=script_name, config=config)
        self.validate = self.config.get("validate", True)

        self.no_rule = []
        self.no_derivation_found = []
        self.not_validated = []
        self.validated = []
        self.not_all_rules_used = []
        self.not_all_nodes_covered = []
        self.parse_did_not_finish = []
        self.cky_did_not_finish = []
        self.all_sens = 0

        self.pos_tag_resolution = False
        self.pred_resolution_from_rule = False

        self.grammar_cuts = self.config.get("grammar_cuts", None)
        self.hrgs = []
        self.grammar_fn_prefix = f"hrg_{self.config_json.split('/')[-1].split('.json')[0].split('train_')[-1]}"

    def _do_for_triplet(self, sen_dir, triplet_idx, triplet_graph_str, triplet):
        hrg_dir = self._get_subdir(str(triplet_idx), self.out_dir)
        triplet_logger = Logger(f"{hrg_dir}/sen{triplet_idx}.log")

        self.all_sens += 1
        triplet_graph = Graph.from_bolinas(triplet_graph_str)
        hrg_for_triplet = self._get_hrg_for_triplet(
            triplet_graph_str, triplet_graph, triplet_idx, triplet, triplet_logger
        )
        if not hrg_for_triplet.all_rules:
            self.no_rule.append(triplet_idx)
            return

        self.hrgs.append(hrg_for_triplet)

        hrg_for_triplet.save_grammar_lines(f"{hrg_dir}/sen{triplet_idx}.hrg")
        triplet_logger.log(f"\nGrammar:\n{hrg_for_triplet.print_grammar_lines()}\n")

        if self.validate:
            grammar = Grammar.load_from_file(
                hrg_for_triplet.get_grammar_lines(),
                VoRule,
                nodelabels=True,
                logprob=True,
            )
            parser = Parser(grammar, stop_at_first=True, permutations=False)

            self.logger.log(f"Parsing sen {triplet_idx}")
            try:
                derivation = parser.check_membership(
                    triplet_graph_str,
                    sen_logger=triplet_logger,
                    global_logger=self.logger,
                    pos_tag_resolution=self.pos_tag_resolution,
                )

                if derivation is None:
                    self.no_derivation_found.append(triplet_idx)
                    return

                triplet_logger.log(f"\nGold triplet:\n{triplet.to_short_json()}\n")
                number_of_used_rule = len(derivation.rules_counter.keys())
                if number_of_used_rule != len(grammar):
                    triplet_logger.log(
                        f"\nNot all rules are used: {number_of_used_rule} of {len(grammar)}\n"
                    )
                    self.not_all_rules_used.append(triplet_idx)

                matcher = TripletMatcher(triplet, derivation.processed_triplet)
                if matcher.exact_match:
                    self.validated.append(triplet_idx)
                else:
                    self.not_validated.append(triplet_idx)
            except ParseTooLongException as e:
                self.parse_did_not_finish.append(triplet_idx)
                triplet_logger.log(e.print_message())
            except CkyTooLongException as e:
                self.cky_did_not_finish.append(triplet_idx)
                triplet_logger.log(e.print_message())
            except NotAllNodesCoveredException as e:
                self.not_all_nodes_covered.append(triplet_idx)
                triplet_logger.log(e.print_message())

    @abstractmethod
    def _get_hrg_for_triplet(
        self, triplet_graph_str, triplet_graph, triplet_id, triplet, triplet_logger
    ):
        raise NotImplemented

    def _after_loop(self):
        if self.validate:
            self.logger.log(
                f"\nNumber of no rules: {len(self.no_rule)}\n"
                f"{json.dumps(self.no_rule)}"
                f"\nNumber of no derivation: {len(self.no_derivation_found)}\n"
                f"{json.dumps(self.no_derivation_found)}"
                f"\nNumber of not validated: {len(self.not_validated)}\n"
                f"{json.dumps(self.not_validated)}"
                f"\nNumber of not all rules used: {len(self.not_all_rules_used)}\n"
                f"{json.dumps(self.not_all_rules_used)}"
                f"\nNumber of not all nodes covered: {len(self.not_all_nodes_covered)}\n"
                f"{json.dumps(self.not_all_nodes_covered)}"
                f"\nNumber of parse did not finish: {len(self.parse_did_not_finish)}\n"
                f"{json.dumps(self.parse_did_not_finish)}"
                f"\nNumber of cky conversion did not finish: {len(self.cky_did_not_finish)}\n"
                f"{json.dumps(self.cky_did_not_finish)}"
                f"\nNumber of validated: {len(self.validated)}"
                f"\nNumber of all sentences: {self.all_sens}\n",
                to_stdout=True,
            )
        self.__save_merged_hrg()
        super()._after_loop()

    def __save_merged_hrg(self):
        self.grammar_dir = self._get_subdir("grammar")

        hrg_for_dataset = HRGForDataset(self.hrgs)
        self.__save_hrg(hrg_for_dataset, f"{self.grammar_fn_prefix}")

        cuts = hrg_for_dataset.get_cuts(self.grammar_cuts)
        for size, hrg_cut in cuts.items():
            self.__save_hrg(hrg_cut, f"{self.grammar_fn_prefix}_{size}")

    def __save_hrg(self, hrg, fn_prefix):
        hrg.save_hrg(self.grammar_dir, fn_prefix)
        self.logger.log(f"\nUnique rules for {fn_prefix}: {hrg.all_rules}")
