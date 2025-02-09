import itertools

from source.common.triplet.triplet import Triplet


class PostProcessedTriplet(Triplet):
    def __init__(self, triplet_dict, pos_tags, top_order, label_to_nodes=True):
        super().__init__(triplet_dict, label_to_nodes)
        self.pos_tags = pos_tags
        self.top_order = top_order

        self.pred_resolution = None
        self.__resolve_pred()

        self.permutations = []

    def __resolve_pred(self):
        preds = [n for n, l in self.node_to_label.items() if l == "P"]
        if len(preds) > 0:
            self.pred_resolution = "X"
            return
        verbs = [n + 1 for n, t in enumerate(self.pos_tags) if t == "VERB"]
        if len(verbs) == 0:
            self.node_to_label[self.top_order[1]] = "P"
            self.pred_resolution = "A"
        elif len(verbs) == 1:
            self.node_to_label[verbs[0]] = "P"
            self.pred_resolution = "B"
        else:
            assert len(verbs) > 1
            first_verb_idx = None
            for v_idx in verbs:
                idx = self.top_order.index(int(v_idx))
                if first_verb_idx is None or idx < first_verb_idx:
                    first_verb_idx = idx
            first_verb_node = self.top_order[first_verb_idx]
            self.node_to_label[first_verb_node] = "P"
            self.pred_resolution = "C"
        self._update_label_to_nodes()

    def calculate_all_permutations(self):
        if not self.permutations:
            args = self.arguments()
            groups = list(args.values())
            permutations = list(itertools.permutations(args.keys()))
            for permutation in permutations:
                label_dict = {"P": self.predicate()}
                for i, arg_idx in enumerate(permutation):
                    label_dict[arg_idx] = groups[i]
                self.permutations.append(Triplet(label_dict))
