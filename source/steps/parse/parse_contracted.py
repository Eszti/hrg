import os

from source.steps.parse.parse import Parse


class ParseComplete(Parse):

    def __init__(self, config=None):
        super().__init__(
            description="Script to parse graph inputs for arg contracted graphs and save parsed chars.",
            script_name="parse",
            config=config,
        )
        self.graph_file_suffix = "_contracted.graph"

    def _get_triplet_input(self, sen_idx, sen_dir):
        graph_files = sorted(
            [
                f"{sen_dir}/{fn}"
                for fn in os.listdir(sen_dir)
                if fn.endswith(self.graph_file_suffix)
            ],
            key=lambda x: self._get_triplet_id(x),
        )

        return [(self._get_triplet_id(fn), fn) for fn in graph_files]


if __name__ == "__main__":
    ParseComplete().run()
