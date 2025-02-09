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

    def _update_label_to_nodes(self):
        label_to_nodes_dict = defaultdict(list)
        for node, label in self.node_to_label.items():
            label_to_nodes_dict[label].append(node)
        self.label_to_nodes = label_to_nodes_dict
        self.__sort_label_to_nodes()

    def __sort_label_to_nodes(self):
        self.label_to_nodes = {
            label: sorted([int(a) for a in args])
            for label, args in self.label_to_nodes.items()
        }

    def to_json_str(self):
        return json.dumps(OrderedDict(sorted(self.label_to_nodes.items())))

    def to_file(self, fn):
        with open(fn, "w") as f:
            f.write(self.to_json_str())

    @staticmethod
    def from_json_str(json_str):
        label_to_nodes_dict = json.loads(json_str)
        return Triplet(label_to_nodes_dict)

    @staticmethod
    def from_file(fn):
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
