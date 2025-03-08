from source.steps.eval.eval import Eval


class EvalContracted(Eval):

    def __init__(self, config=None):
        super().__init__(
            description="Script to evaluate complete systems.",
            script_name="eval_complete",
            config=config,
        )


if __name__ == "__main__":
    EvalContracted().run()
