from datetime import datetime

from source.common.script.script import Script
from source.steps.eval.eval import Eval
from source.steps.kbest.kbest import KBest
from source.steps.parse.parse import Parse
from source.steps.predict.merge import Merge
from source.steps.preproc.preproc import Preproc
from source.steps.random.artefacts import Artefacts
from source.steps.random.random_extractor import Random
from source.steps.stat.run_all_stat import Stat
from source.steps.train.hrg import Hrg
from source.steps.train.train import Train


class Pipeline(Script):
    def __init__(self, log=True, config=None):
        super().__init__(
            "Script to run a pipeline.", log=log, script_name="pipeline", config=config
        )
        self.steps = self.config["steps"]
        self.name_to_class = {
            "preproc": Preproc,
            "train": Train,
            "hrg": Hrg,
            "artefacts": Artefacts,
            "random": Random,
            "parse": Parse,
            "kbest": KBest,
            "merge": Merge,
            "eval": Eval,
            "stat": Stat,
        }

    def _run_loop(self):
        for step in self.steps:
            step_name = step["step_name"]
            script_name = step["script_name"]
            self.logger.log(f"Processing step {step_name}: {datetime.now()}")
            step_class = self.name_to_class[script_name]
            config = f"{self.pipeline_dir}/config/{step['config']}"
            step = step_class(config=config)
            if self.first is not None:
                step.first = self.first
            if self.last is not None:
                step.last = self.last
            step.run()


if __name__ == "__main__":
    Pipeline().run()
