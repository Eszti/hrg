import argparse
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime

from source.common.script.logger import Logger


class Script(ABC):
    def __init__(self, description, log, config=None):
        args = self._get_data_dir_and_config_args(description)
        if config is None:
            self.config_json = args.config
        else:
            self.config_json = config
        self.data_dir = args.data_dir
        self.pipeline_dir = os.path.dirname(
            os.path.dirname(os.path.realpath(self.config_json))
        )
        self.script_output_root = f"{self.pipeline_dir}/output"
        self.config_name = self.config_json.split("/")[-1].split(".json")[0]
        self.config = json.load(open(self.config_json))
        self._setup_logger(log)
        self.first_sen_to_proc = None
        self.last_sen_to_proc = None
        self.out_dir = (
            f"{self.data_dir}/{self.config['out_dir']}"
            if "out_dir" in self.config
            else None
        )
        self.first = self.config.get("first", None)
        self.last = self.config.get("last", None)

    def _setup_logger(self, log):
        if log:
            self.logger = Logger(
                log_file=f"{self._get_subdir('log', parent_dir=self.pipeline_dir)}/{self.config_name}.log"
            )
            self.start_time = time.time()
            self.logger.log(
                f"Execution start: {datetime.now()}\n{json.dumps(self.config, indent=4)}\n"
            )

    def run(self):
        self._before_loop()
        self._run_loop()
        self._after_loop()

    def _before_loop(self):
        pass

    @abstractmethod
    def _run_loop(self):
        raise NotImplemented

    def _after_loop(self):
        if self.logger:
            if self.first_sen_to_proc is not None or self.last_sen_to_proc is not None:
                self.logger.log(
                    f"\nFirst sentence to process: {self.first_sen_to_proc}"
                    f"\nLast sentence to process: {self.last_sen_to_proc}",
                    to_stdout=True,
                )
            self.logger.log(f"\nExecution finish: {datetime.now()}", to_stdout=True)
            elapsed_time = time.time() - self.start_time
            self.logger.log(
                f"Elapsed time: {round(elapsed_time / 60)} min {round(elapsed_time % 60)} sec\n",
                to_stdout=True,
            )

    def _get_subdir(self, name, parent_dir=None, create=True):
        if parent_dir is None:
            parent_dir = self.script_output_root
        subdir = f"{parent_dir}/{name}"
        if not os.path.exists(subdir):
            if create:
                os.makedirs(subdir)
            else:
                raise RuntimeError(f"{subdir} does not exist")
        return subdir

    @staticmethod
    def _add_filter_and_postprocess(name, chart_filter, postprocess, delim="/"):
        if chart_filter:
            name += f"{delim}{chart_filter}"
        if postprocess:
            name += f"{delim}{postprocess}"
        return name

    @staticmethod
    def _get_data_dir_and_config_args(desc=""):
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument("-d", "--data-dir", type=str)
        parser.add_argument("-c", "--config", type=str, default="")
        return parser.parse_args()
