from source.steps.bolinas.bolinas import Bolinas
from source.steps.parse.parse_complete import ParseCompleteStep


class BolinasComplete(Bolinas):
    def __init__(self, config=None):
        super().__init__(
            description=None,
            script_name="bolinas_complete",
            config=config,
        )

    def _get_triplet_input(self, sen_idx, sen_dir):
        return ParseCompleteStep.get_triplet_input(sen_idx, sen_dir)


if __name__ == "__main__":
    BolinasComplete().run()
