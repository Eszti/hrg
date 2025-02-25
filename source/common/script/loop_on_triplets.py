import os
from abc import abstractmethod

from source.common.script.loop_on_sen_dirs import LoopOnSenDirs
from source.common.triplet.triplets_for_sen import TripletsForSen


class LoopOnTriplets(LoopOnSenDirs):
    def __init__(self, description, script_name, config):
        super().__init__(description, script_name=script_name, log=True, config=config)
        self.first_triplet = self.config.get("first_triplet", None)
        self.last_triplet = self.config.get("last_triplet", None)
        self.graph_file_suffix = "_triplet.graph"
        self.gold_triplets_fn = "gold_triplets.json"

    def _do_for_sen(self, sen_idx, sen_dir):
        graph_files = sorted(
            [
                f"{sen_dir}/{fn}"
                for fn in os.listdir(sen_dir)
                if fn.endswith(self.graph_file_suffix)
            ],
            key=lambda x: int(x.split("/")[-1].split("_")[0].split("sen")[-1]),
        )
        if not graph_files:
            return

        triplets = TripletsForSen.from_json(
            f"{sen_dir}/{self.gold_triplets_fn}"
        ).triplets
        for i, graph_file in enumerate(graph_files):
            triplet_idx = int(
                graph_file.split("/")[-1]
                .split(self.graph_file_suffix)[0]
                .split("sen")[-1]
            )
            if self.first_triplet is not None and triplet_idx < self.first_triplet:
                continue
            if self.last_triplet is not None and triplet_idx == self.last_triplet:
                self.break_loop = True
            print(f"\nProcessing triplet {triplet_idx}")
            with open(graph_file) as f:
                lines = f.readlines()
                assert len(lines) == 1
                triplet_graph_str = lines[0].strip()
            triplet_candidates = [t for t in triplets if t.triplet_id == triplet_idx]
            assert len(triplet_candidates) == 1
            self._do_for_triplet(
                sen_dir, triplet_idx, triplet_graph_str, triplet_candidates[0]
            )
            if self.break_loop:
                break

    @abstractmethod
    def _do_for_triplet(self, sen_dir, triplet_idx, triplet_graph_str, triplet):
        raise NotImplemented
