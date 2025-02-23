import os
from abc import abstractmethod
from collections import defaultdict

from source.common.scores.sentence_scorer import SentenceScorer
from source.common.scores.system_scorer import SystemScorer
from source.common.script.loop_on_sen_dirs import LoopOnSenDirs
from source.common.triplet.triplets_for_sen import TripletsForSen


class Eval(LoopOnSenDirs):
    def __init__(self, description, script_name, config=None):
        super().__init__(description, script_name=script_name, config=config)
        self.grammar_dirs = self.config["grammar_dirs"]
        self.models = self.config["models"]
        self.strict_match = self.config.get("strict_match", True)
        self.sentence_scorers = defaultdict(lambda: defaultdict(list))
        self.match_ids = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.report_dir = self._get_subdir("eval")

    def _do_for_sen(self, sen_idx, preproc_sen_dir):
        for grammar_dir in self.grammar_dirs:
            sen_dir = f"{self.data_dir}/{grammar_dir}/{str(sen_idx)}"
            kbest_dir = f"{sen_dir}/kbest"
            out_dir = self._get_subdir("eval", parent_dir=sen_dir)

            triplets_fn = self._get_gold_triplets_fn(preproc_sen_dir)
            if triplets_fn is None:
                continue
            gold_triplets_for_sen = TripletsForSen.from_json(triplets_fn)

            for model_file in [
                fn for fn in os.listdir(kbest_dir) if fn.endswith("_triplets.json")
            ]:
                model_name_form_file = model_file.split("_")[1]
                model_names = self.__get_model_names_from_config(model_name_form_file)
                for model_name in model_names:
                    predicted_triplets_for_sen = TripletsForSen.from_json(
                        f"{kbest_dir}/{model_file}"
                    )
                    if model_name.startswith("basic") or model_name.startswith("max"):
                        arg_perm = model_name.endswith("_ap")
                        for k in [1, 3, 5, 10]:
                            first_k_triplets_for_sen = TripletsForSen(
                                predicted_triplets_for_sen.triplets[:k],
                                predicted_triplets_for_sen.sen_id,
                                predicted_triplets_for_sen.sen_text,
                            )
                            for t in first_k_triplets_for_sen.triplets:
                                assert t.triplet_id <= k
                            self.__calculate_scores(
                                grammar_dir,
                                f"{model_name}_{k}",
                                gold_triplets_for_sen,
                                first_k_triplets_for_sen,
                                out_dir,
                                sen_idx,
                                arg_perm,
                            )
                    else:
                        assert (
                            model_name == "f1"
                            or model_name == "prec"
                            or model_name == "rec"
                        )
                        self.__calculate_scores(
                            grammar_dir,
                            model_name,
                            gold_triplets_for_sen,
                            predicted_triplets_for_sen,
                            out_dir,
                            sen_idx,
                            False,
                        )

    @abstractmethod
    def _get_gold_triplets_fn(self, preproc_sen_dir):
        raise NotImplemented

    def __get_model_names_from_config(self, model_name_form_file):
        model_name_candidates = [model_name_form_file, f"{model_name_form_file}_ap"]
        return [mn for mn in model_name_candidates if mn in self.models]

    def __calculate_scores(
        self,
        grammar_dir,
        model_name,
        gold_triplets_for_sen,
        predicted_triplets_for_sen,
        out_dir,
        sen_idx,
        arg_perm,
    ):
        sentence_scorer = SentenceScorer(
            gold_triplets_for_sen,
            predicted_triplets_for_sen,
            arg_perm=arg_perm,
            strict=self.strict_match,
        )
        sentence_scorer.to_file(f"{out_dir}/sen{sen_idx}_{model_name}_scores.txt")
        self.sentence_scorers[grammar_dir][model_name].append(sentence_scorer)
        if sentence_scorer.matches:
            for i in range(len(sentence_scorer.matches)):
                self.match_ids[grammar_dir][model_name]["matches"].append(sen_idx)
        if sentence_scorer.exact_matches:
            for i in range(len(sentence_scorer.exact_matches)):
                self.match_ids[grammar_dir][model_name]["exact_matches"].append(sen_idx)

    def _after_loop(self):
        sys_scorer = SystemScorer(self.sentence_scorers)
        sys_scorer.log_results(self.logger)
        sys_scorer.save_report(f"{self.report_dir}/{self.config_name}.md")
        with open(f"{self.report_dir}/matches_{self.config_name}.txt", "w") as f:
            for grammar_dir, matches_for_grammar in sorted(self.match_ids.items()):
                f.write(f"\n{grammar_dir}\n")
                for model_name, matches in sorted(matches_for_grammar.items()):
                    f.write(f"\n{model_name}\n")
                    for match_name, ids in matches.items():
                        f.write(f"{match_name}\n{ids}\n")
        super()._after_loop()
