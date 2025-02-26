from source.steps.eval.eval import Eval


class EvalContracted(Eval):

    def __init__(self, config=None):
        super().__init__(
            description="Script to evaluate contracted systems.",
            script_name="eval_complete",
            config=config,
        )
        self.gold_triplet_fn = "gold_contracted_triplets.json"


if __name__ == "__main__":
    EvalContracted().run()
