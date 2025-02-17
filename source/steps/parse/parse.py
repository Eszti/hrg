import fileinput
import json

from source.common.bolinas.cky_chart import CkyChart
from source.common.bolinas.hgraph import Hgraph
from source.common.bolinas.parser import Parser
from source.common.exceptions import ParseTooLongException, CkyTooLongException
from source.common.script.logger import Logger
from source.common.script.loop_on_sen_dirs import LoopOnSenDirs


class Parse(LoopOnSenDirs):

    def __init__(self, config=None):
        super().__init__(
            description="Script to parse graph inputs and save parsed chars.",
            script_name="parse",
            config=config,
        )
        self.grammar = None
        self.parser = None
        self.parse_did_not_finish = []
        self.cky_did_not_finish = []

    def _before_loop(self):
        self._load_grammar()
        self.parser = Parser(
            self.grammar, max_steps=self.config.get("max_steps", 10000)
        )

    def _do_for_sen(self, sen_idx, sen_dir):
        bolinas_dir = self._get_subdir(
            "parse", parent_dir=f"{self.out_dir}/{str(sen_idx)}"
        )
        sen_logger = Logger(f"{bolinas_dir}/sen{str(sen_idx)}_parse.log")
        try:
            self._parse_sen(
                graph_file=f"{sen_dir}/pos_edge.graph",
                chart_file=f"{bolinas_dir}/sen{str(sen_idx)}_chart.pickle",
                sen_logger=sen_logger,
            )
        except ParseTooLongException as e:
            self.parse_did_not_finish.append(sen_idx)
            sen_logger.log(e.print_message())
        except CkyTooLongException as e:
            self.cky_did_not_finish.append(sen_idx)
            sen_logger.log(e.print_message())

    def _parse_sen(self, graph_file, chart_file, sen_logger):
        parse_generator = self.parser.parse_graphs(
            (Hgraph.from_string(x) for x in fileinput.FileInput(graph_file)),
            partial=True,
            logger=sen_logger,
        )

        for i, cky_chart in enumerate(parse_generator):
            assert i == 0
            if cky_chart.no_derivation():
                sen_logger.log("No derivation found")
                continue
            else:
                CkyChart.to_file(cky_chart, chart_file)

    def _after_loop(self):
        self.logger.log(
            f"\nNumber of parse did not finish: {len(self.parse_did_not_finish)}\n"
            f"{json.dumps(self.parse_did_not_finish)}"
            f"\nNumber of cky conversion did not finish: {len(self.cky_did_not_finish)}\n"
            f"{json.dumps(self.cky_did_not_finish)}",
        )
        super()._after_loop()


if __name__ == "__main__":
    Parse().run()
