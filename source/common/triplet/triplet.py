import itertools
import json
from collections import defaultdict, OrderedDict


class Triplet:
    def __init__(
        self,
        triplet_dict,
        triplet_id="",
        pred_resolution=None,
        derivation_score=0,
        label_to_nodes=True,
    ):
        if label_to_nodes:
            self.label_to_nodes = triplet_dict
            self.__sort_label_to_nodes()
            self.__update_node_to_label()
        else:
            self.node_to_label = {
                int(node): label for node, label in triplet_dict.items()
            }
            self.__update_label_to_nodes()
        self.triplet_id = triplet_id
        self.pred_resolution = pred_resolution
        self.derivation_score = derivation_score

    def __update_label_to_nodes(self):
        label_to_nodes_dict = defaultdict(list)
        for node, label in self.node_to_label.items():
            label_to_nodes_dict[label].append(node)
        self.label_to_nodes = label_to_nodes_dict
        self.__sort_label_to_nodes()

    def __update_node_to_label(self):
        self.node_to_label = {
            int(n): label for label, nodes in self.label_to_nodes.items() for n in nodes
        }

    def __sort_label_to_nodes(self):
        self.label_to_nodes = OrderedDict(
            sorted(
                {
                    label: sorted([int(a) for a in args])
                    for label, args in self.label_to_nodes.items()
                }.items()
            )
        )
        self.__assert_labels()

    def __assert_labels(self):
        for label, nodes in self.label_to_nodes.items():
            assert label.startswith("P") or label.startswith("A")

    def to_dict(self):
        json_dict = dict()
        json_dict["labels"] = self.label_to_nodes
        if self.triplet_id is not None:
            json_dict["triplet_id"] = self.triplet_id
        if self.pred_resolution is not None:
            json_dict["pred_resolution"] = self.pred_resolution
        if self.derivation_score is not None:
            json_dict["derivation_score"] = self.derivation_score
        return json_dict

    def to_short_json(self):
        return f"{self.triplet_id};{json.dumps(self.label_to_nodes)};{self.derivation_score:g} - len: {self.len()}"

    def get_label(self, node):
        return self.node_to_label.get(node)

    def arguments(self):
        return {k: v for k, v in self.label_to_nodes.items() if k.startswith("A")}

    def predicate(self):
        return self.label_to_nodes.get("P", [])

    def a0(self):
        return self.label_to_nodes.get("A0", [])

    def len(self):
        ret = 0
        for label, nodes in self.label_to_nodes.items():
            ret += len(nodes)
        return ret

    def resolve_pred(self, pos_tags, top_order, pos_tag_resolution=False):
        predicates = [n for n, l in self.node_to_label.items() if l == "P"]
        if (len(predicates) > 0 and not pos_tag_resolution) or (
            len(predicates) == 1 and pos_tag_resolution
        ):
            self.pred_resolution = "X"
            return
        if len(predicates) > 1 and pos_tag_resolution:
            pred_top_order = [n for n in top_order if n in predicates]
            self.label_to_nodes["P"] = [pred_top_order[0]]
            self.__update_node_to_label()
            self.pred_resolution = "D"
        elif len(predicates) == 0:
            verbs = [n + 1 for n, t in enumerate(pos_tags) if t == "VERB"]
            if len(verbs) == 0:
                self.node_to_label[top_order[1]] = "P"
                self.pred_resolution = "A"
            elif len(verbs) == 1:
                self.node_to_label[verbs[0]] = "P"
                self.pred_resolution = "B"
            else:
                assert len(verbs) > 1
                first_verb_idx = None
                for v_idx in verbs:
                    idx = top_order.index(int(v_idx))
                    if first_verb_idx is None or idx < first_verb_idx:
                        first_verb_idx = idx
                first_verb_node = top_order[first_verb_idx]
                self.node_to_label[first_verb_node] = "P"
                self.pred_resolution = "C"
        self.__update_label_to_nodes()

    def get_all_permutations(self):
        ret = []
        args = self.arguments()
        groups = list(args.values())
        permutations = list(itertools.permutations(args.keys()))
        for permutation in permutations:
            label_dict = {"P": self.predicate()}
            for i, arg_idx in enumerate(permutation):
                label_dict[arg_idx] = groups[i]
            ret.append(
                Triplet(
                    label_dict,
                    self.triplet_id,
                    self.pred_resolution,
                    self.derivation_score,
                )
            )
        return ret
