from source.common.bolinas.hgraph import Hgraph


class HRGRule:
    def __init__(self, lhs, rhs_string, triplet_id, weight=0):
        self.lhs = lhs
        self.rhs = rhs_string
        self.triplet_id = triplet_id
        self.weight = weight

    def print_rule(self):
        return f"{self.get_rule_string()};{self.weight}"

    def get_rule_string(self):
        return f"{self.lhs} -> {self.rhs}"

    def __get_rhs_string(self, rhs_string):
        rhs_from_string = Hgraph.from_string(rhs_string)
        self.rhs = Hgraph.from_string(rhs_from_string.to_bolinas_str(nodeids=True))
