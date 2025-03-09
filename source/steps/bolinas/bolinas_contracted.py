from source.steps.bolinas.bolinas import Bolinas
from source.steps.parse.parse_contracted import ParseContractedStep


class BolinasContracted(Bolinas):
    def __init__(self, config=None):
        super().__init__(
            description=None,
            script_name="bolinas_contracted",
            config=config,
        )
        self.pos_tag_resolution = True
        self.pred_resolution_from_rule = True
        self.arg_perm = False
        self.gold_triplets_fn = "gold_contracted_triplets.json"

    def _get_triplet_input(self, sen_idx, sen_dir):
        return ParseContractedStep.get_triplet_input(sen_idx, sen_dir)


if __name__ == "__main__":
    BolinasContracted().run()
