import json
from abc import abstractmethod

from source.common.bolinas.cky_chart import CkyChart
from source.common.bolinas.hgraph import Hgraph
from source.common.bolinas.parser import Parser
from source.common.exceptions import ParseTooLongException, CkyTooLongException
from source.common.script.logger import Logger
from source.common.script.loop_on_sen_dirs import LoopOnSenDirs


class ParseStep:
    def __init__(self, parser, logger):
        self.parser = parser
        self.logger = logger
        self.parse_did_not_finish = []
        self.cky_did_not_finish = []
        self.parse_finished = []
        self.no_derivation = []

    def parse_sen(self, triplet_id, graph_file, sen_logger):
        try:
            cky_chart = self.__parse(
                triplet_id=triplet_id,
                graph_file=f"{graph_file}",
                global_logger=self.logger,
                sen_logger=sen_logger,
            )
            self.parse_finished.append(triplet_id)
            return cky_chart
        except ParseTooLongException as e:
            self.parse_did_not_finish.append(triplet_id)
            sen_logger.log(e.print_message())
            self.logger.log(e.print_message())
        except CkyTooLongException as e:
            self.cky_did_not_finish.append(triplet_id)
            sen_logger.log(e.print_message())
            self.logger.log(e.print_message())
        return None

    def __parse(self, triplet_id, graph_file, global_logger, sen_logger):
        parse_generator = self.parser.parse_graphs(
            (Hgraph.from_string(x) for x in self.__read_graph_file(graph_file)),
            partial=True,
            sen_logger=sen_logger,
            global_logger=global_logger,
        )
        for i, cky_chart in enumerate(parse_generator):
            assert i == 0
            if cky_chart.no_derivation():
                sen_logger.log("\nNo derivation found")
                self.no_derivation.append(triplet_id)
            else:
                return cky_chart

    @staticmethod
    def __read_graph_file(graph_file):
        with open(graph_file, "r") as f:
            lines = f.readlines()
        return lines

    def log_parse_step(self):
        len_parse_dnf = len(self.parse_did_not_finish)
        len_cky_dnf = len(self.cky_did_not_finish)
        len_parse_success = len(self.parse_finished)
        all_sens = len_parse_success + len_parse_dnf + len_cky_dnf
        self.logger.log(
            f"\nNumber of parse did not finish: {len_parse_dnf}\n"
            f"{json.dumps(self.parse_did_not_finish)}"
            f"\nNumber of cky conversion did not finish: {len_cky_dnf}\n"
            f"{json.dumps(self.cky_did_not_finish)}"
            f"\nNumber of parse finished: {len_parse_success}\n"
            f"\nNumber of no derivation found: {len(self.no_derivation)}\n"
            f"{json.dumps(self.no_derivation)}\n"
            f"\nAll sentences: {all_sens}"
            f"\nFinished: {round(len_parse_success / all_sens, 2)}"
            f"\nDNF: {round((len_parse_dnf + len_cky_dnf) / all_sens, 2)}",
            to_stdout=True,
        )


class Parse(LoopOnSenDirs):
    def __init__(self, description, script_name, config=None):
        super().__init__(description, script_name=script_name, config=config)
        self.grammar = None

    def _before_loop(self):
        self._load_grammar()
        parser = Parser(self.grammar, max_steps=self.config.get("max_steps"))
        self.parse_step = ParseStep(parser, self.logger)

    def _do_for_sen(self, sen_idx, sen_dir):
        for triplet_id, graph_file in self._get_triplet_input(sen_idx, sen_dir):
            bolinas_dir = self._get_subdir(
                "parse", parent_dir=f"{self.out_dir}/{str(triplet_id)}"
            )
            sen_logger = Logger(f"{bolinas_dir}/sen{str(triplet_id)}_parse.log")
            self.logger.log(f"\nParsing sen {triplet_id}")
            cky_chart = self.parse_step.parse_sen(
                triplet_id=triplet_id,
                graph_file=graph_file,
                sen_logger=sen_logger,
            )
            if cky_chart is not None:
                chart_file = f"{bolinas_dir}/sen{str(triplet_id)}_chart.pickle"
                CkyChart.to_file(cky_chart, chart_file)

    @abstractmethod
    def _get_triplet_input(self, sen_idx, sen_dir):
        raise NotImplemented

    def _after_loop(self):
        self.parse_step.log_parse_step()
        super()._after_loop()
