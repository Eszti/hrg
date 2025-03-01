class HRGForTriplet:
    def __init__(self, hrg_rules, triplet_id):
        self.triplet_id = triplet_id
        self.all_rules = hrg_rules
        self.initial_rules = []
        self.other_rules = []
        self.__fill_split_rules()
        self.__set_predicate_mapping_failure()

    def __fill_split_rules(self):
        for rule in self.all_rules:
            if rule.lhs == "S":
                self.initial_rules.append(rule)
            else:
                self.other_rules.append(rule)

    def __set_predicate_mapping_failure(self):
        self.failed_predicate_mapping = False
        for rule in self.all_rules:
            if rule.failed_predicate_mapping:
                self.failed_predicate_mapping = True
                return

    def get_grammar_lines(self):
        ret = []
        for rule in self.initial_rules:
            ret.append(rule.print_rule())
        for rule in self.other_rules:
            ret.append(rule.print_rule())
        return ret

    def print_grammar_lines(self):
        ret = ""
        for line in self.get_grammar_lines():
            ret += f"{line}\n"
        return ret

    def save_grammar_lines(self, fn):
        with open(fn, "w") as f:
            f.write(self.print_grammar_lines())
