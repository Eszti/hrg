class KbestModel:
    def __init__(self):
        self.max_filter = False
        self.kbest = True
        self.k = 10
        self.subdir = None

    def get_derivation_per_model(self, derivation_list, gold_triplets, pos_tags, top_order, arg_perm=False):
        if self.kbest:
            return {self.subdir: derivation_list.get_k_best_unique_derivation(self.k)}
        return derivation_list.get_best_matching_derivations(gold_triplets, pos_tags, top_order, arg_perm)

class BasicModel(KbestModel):
    def __init__(self):
        super().__init__()
        self.subdir = "basic"

class MaxModel(KbestModel):
    def __init__(self):
        super().__init__()
        self.subdir = "max"
        self.max_filter = True

class PRModel(KbestModel):
    def __init__(self):
        super().__init__()
        self.subdir = "pr_best"
        self.kbest = False