from source.common.script.script import Script
from source.steps.stat.k_stat import KStat
from source.steps.stat.pred_eval import PredEval
from source.steps.stat.rule_stat import RuleStat


class Stat(Script):
    def __init__(self, config=None):
        super().__init__(
            description="Script to run all dev statistics at once.",
            script_name="run_all_stat",
            config=config,
            log=False,
        )
        self.config_dir = self._get_subdir(
            "config", parent_dir=self.pipeline_dir, create=False
        )

    def _run_loop(self):
        print("Calculate k stat")
        KStat(f"{self.config_dir}/{self.config['k_stat_config']}").run()
        print("Evaluate predicate resolution")
        PredEval(f"{self.config_dir}/{self.config['pred_eval_config']}").run()
        print("Calculate rule stat")
        RuleStat(f"{self.config_dir}/{self.config['rule_stat_config']}").run()


if __name__ == "__main__":
    Stat().run()
