from collections import defaultdict, Counter


class HRGForDataset:
    def __init__(self, hrgs):
        self.hrgs = hrgs
        self.duplicate_rules = []
        self.nt_to_rules = defaultdict(lambda: defaultdict(list))
        self.nt_to_rules_counter = defaultdict(Counter)
        self.weighted_grammar_lines = defaultdict(list)
        self.all_rules = 0
        self.__calculate_variables()

    def __calculate_variables(self):
        duplicate_collector = defaultdict(lambda: defaultdict(list))
        for hrg_for_triplet in self.hrgs:
            for hrg_rule in hrg_for_triplet.all_rules:
                self.nt_to_rules[hrg_rule.lhs][
                    hrg_rule.get_rule_and_predicate_ids()
                ].append(hrg_for_triplet)
                duplicate_collector[hrg_rule.get_rule_string()][
                    hrg_rule.get_predicate_ids()
                ].append(hrg_rule.triplet_id)
        for rule, predicates in duplicate_collector.items():
            if len(predicates) > 1:
                for predicate_ids, triplet_ids in predicates.items():
                    self.duplicate_rules.append(
                        f"{rule};{predicate_ids}; {triplet_ids}"
                    )
        for nt, unique_rule_strings_dict in self.nt_to_rules.items():
            for (
                unique_rule_string,
                hrgs_for_triplet,
            ) in unique_rule_strings_dict.items():
                self.nt_to_rules_counter[nt][unique_rule_string] = len(hrgs_for_triplet)
        for nt, unique_rule_strings_counter in self.nt_to_rules_counter.items():
            for unique_rule_string, cnt in unique_rule_strings_counter.most_common():
                w = round(cnt / unique_rule_strings_counter.total(), 2)
                if w < 0.01:
                    w = 0.01
                self.weighted_grammar_lines[nt].append((unique_rule_string, cnt, w))
                self.all_rules += 1

    def log_duplicates_with_different_predicates(self):
        ret = ""
        for line in self.duplicate_rules:
            ret += f"{line}\n"
        return ret

    def save_hrg(self, grammar_dir, grammar_fn_prefix):
        lines = self.__get_grammar_lines()
        with open(f"{grammar_dir}/{grammar_fn_prefix}.hrg", "w") as f:
            f.writelines(lines["weight"])
        with open(f"{grammar_dir}/{grammar_fn_prefix}.stat", "w") as f:
            f.writelines(lines["cnt"])

    def get_cuts(self, cuts):
        new_hrgs = {}
        if cuts:
            for size in cuts:
                factor = size / self.all_rules
                hrgs_to_keep = []
                for nt, rules in self.nt_to_rules.items():
                    rules_counter = Counter()
                    for rule_str, hrgs in rules.items():
                        rules_counter[rule_str] = len(hrgs)
                    for rule_str, cnt in rules_counter.most_common(
                        n=int(round(factor * len(rules_counter)))
                    ):
                        hrgs_to_add = rules[rule_str]
                        assert len(hrgs_to_add) == cnt
                        hrgs_to_keep += hrgs_to_add
                new_hrgs[f"{size}"] = HRGForDataset(hrgs_to_keep)
        return new_hrgs

    def __get_grammar_lines(self):
        lines = defaultdict(list)
        for unique_rule_str, cnt, w in self.weighted_grammar_lines["S"]:
            lines["cnt"].append(f"{unique_rule_str};\t{cnt}\n")
            lines["weight"].append(f"{unique_rule_str};\t{w}\n")
        for nt, unique_rule_strings in self.weighted_grammar_lines.items():
            if nt == "S":
                continue
            for unique_rule_str, cnt, w in unique_rule_strings:
                lines["cnt"].append(f"{unique_rule_str};\t{cnt}\n")
                lines["weight"].append(f"{unique_rule_str};\t{w}\n")
        assert len(lines["cnt"]) == len(lines["weight"]) == self.all_rules
        return lines
