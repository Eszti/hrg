import json
from collections import defaultdict, OrderedDict


class Triplet:
    def __init__(self, triplet_dict, label_to_nodes=True):
        if label_to_nodes:
            self.label_to_nodes = triplet_dict
            self.__sort_label_to_nodes()
            self.node_to_label = {
                int(n): label
                for label, nodes in self.label_to_nodes.items()
                for n in nodes
            }
        else:
            self.node_to_label = {
                int(node): label for node, label in triplet_dict.items()
            }
            self._update_label_to_nodes()
        self.pred_resolution = None
        self.derivation_score = None
        self.k = None

    @staticmethod
    def from_processed_triplet(processed_triplet, score, k):
        if processed_triplet.best_permutation is None:
            node_to_label = processed_triplet.node_to_label
        else:
            node_to_label = processed_triplet.best_permutation.node_to_label
        new_triplet = Triplet(node_to_label, label_to_nodes=False)
        new_triplet.pred_resolution = processed_triplet.pred_resolution
        new_triplet.derivation_score = score
        new_triplet.k = k
        return new_triplet

    def _update_label_to_nodes(self):
        label_to_nodes_dict = defaultdict(list)
        for node, label in self.node_to_label.items():
            label_to_nodes_dict[label].append(node)
        self.label_to_nodes = label_to_nodes_dict
        self.__sort_label_to_nodes()

    def __sort_label_to_nodes(self):
        self.label_to_nodes = OrderedDict(
            sorted(
                {
                    label: sorted([int(a) for a in args])
                    for label, args in self.label_to_nodes.items()
                }.items()
            )
        )

    def to_dict(self):
        json_dict = dict()
        json_dict["labels"] = self.label_to_nodes
        json_dict["pred_resolution"] = self.pred_resolution
        json_dict["derivation_score"] = self.derivation_score
        json_dict["k"] = self.k
        return json_dict

    def to_short_json(self):
        return json.dumps(self.label_to_nodes)

    def to_short_file(self, fn):
        with open(fn, "w") as f:
            f.write(self.to_short_json())

    @staticmethod
    def from_short_file(fn):
        label_to_nodes_dict = json.load(open(fn))
        return Triplet(label_to_nodes_dict)

    def get_label(self, node):
        return self.node_to_label.get(node)

    def arguments(self):
        return {k: v for k, v in self.label_to_nodes.items() if k.startswith("A")}

    def predicate(self):
        return self.label_to_nodes["P"]

    def match(self, other):
        p_match = set(self.label_to_nodes["P"]) & set(other.label_to_nodes["P"])
        a0_match = set(self.label_to_nodes["A0"]) & set(other.label_to_nodes["A0"])
        return p_match and a0_match

    def len(self):
        ret = 0
        for label, nodes in self.label_to_nodes.items():
            if label.startswith("P") or label.startswith("A"):
                ret += len(nodes)
        return ret
