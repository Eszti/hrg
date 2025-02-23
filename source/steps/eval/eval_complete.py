from source.steps.eval.eval import Eval


class EvalComplete(Eval):

    def __init__(self, config=None):
        super().__init__(
            description="Script to evaluate systems.",
            script_name="eval_complete",
            config=config,
        )

    def _get_gold_triplets_fn(self, preproc_sen_dir):
        return f"{preproc_sen_dir}/gold_triplets.json"


if __name__ == "__main__":
    EvalComplete().run()
