import os

from source.steps.eval.eval import Eval


class EvalContracted(Eval):

    def __init__(self, config=None):
        super().__init__(
            description="Script to evaluate contracted systems.",
            script_name="eval_complete",
            config=config,
        )
        self.gold_triplet_fn_suffix = "gold_contracted_triplets.json"

    def _get_gold_triplets_fn(self, preproc_sen_dir):
        gold_triplet_files = sorted(
            [
                f"{preproc_sen_dir}/{fn}"
                for fn in os.listdir(preproc_sen_dir)
                if fn.endswith(self.gold_triplet_fn_suffix)
            ],
            key=lambda x: self._get_triplet_id(x),
        )
        assert len(gold_triplet_files) <= 1
        return gold_triplet_files[0] if len(gold_triplet_files) == 1 else None


if __name__ == "__main__":
    EvalContracted().run()
