import os

from source.common.script.loop_on_sen_dirs import LoopOnSenDirs
from source.steps.parse.parse import Parse


class ParseContractedStep:
    @staticmethod
    def get_triplet_input(sen_idx, sen_dir):
        graph_file_suffix = "_contracted.graph"
        graph_files = sorted(
            [
                f"{sen_dir}/{fn}"
                for fn in os.listdir(sen_dir)
                if fn.endswith(graph_file_suffix)
            ],
            key=lambda x: LoopOnSenDirs.get_triplet_id(x),
        )

        return [(LoopOnSenDirs.get_triplet_id(fn), fn) for fn in graph_files]


class ParseComplete(Parse):

    def __init__(self, config=None):
        super().__init__(
            description="Script to parse graph inputs for arg contracted graphs and save parsed chars.",
            script_name="parse",
            config=config,
        )

    def _get_triplet_input(self, sen_idx, sen_dir):
        return ParseContractedStep.get_triplet_input(sen_idx, sen_dir)


if __name__ == "__main__":
    ParseComplete().run()
