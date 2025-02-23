from source.steps.parse.parse import Parse


class ParseComplete(Parse):

    def __init__(self, config=None):
        super().__init__(
            description="Script to parse graph inputs and save parsed chars.",
            script_name="parse_complete",
            config=config,
        )

    def _get_triplet_input(self, sen_idx, sen_dir):
        return [(sen_idx, f"{sen_dir}/pos_edge.graph")]


if __name__ == "__main__":
    ParseComplete().run()
