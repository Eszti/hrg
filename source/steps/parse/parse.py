import fileinput

from source.common.bolinas.cky_chart import CkyChart
from source.common.bolinas.hgraph import Hgraph
from source.common.bolinas.parser import Parser
from source.common.script.logger import Logger
from source.common.script.loop_on_sen_dirs import LoopOnSenDirs


class Parse(LoopOnSenDirs):

    def __init__(self, config=None):
        super().__init__(
            description="Script to parse graph inputs and save parsed chars.",
            config=config,
        )
        self.grammar = None
        self.parser = None

    def _before_loop(self):
        self._load_grammar()
        self.parser = Parser(
            self.grammar, max_steps=self.config.get("max_steps", 10000)
        )

    def _do_for_sen(self, sen_idx, sen_dir):
        bolinas_dir = self._get_subdir(
            "parse", parent_dir=f"{self.out_dir}/{str(sen_idx)}"
        )
        self._parse_sen(
            graph_file=f"{sen_dir}/pos_edge.graph",
            chart_file=f"{bolinas_dir}/sen{str(sen_idx)}_chart.pickle",
            sen_log_file=f"{bolinas_dir}/sen{str(sen_idx)}_parse.log",
        )

    def _parse_sen(self, graph_file, chart_file, sen_log_file):
        sen_logger = Logger(sen_log_file)
        parse_generator = self.parser.parse_graphs(
            (Hgraph.from_string(x) for x in fileinput.input(graph_file)),
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


if __name__ == "__main__":
    Parse().run()
