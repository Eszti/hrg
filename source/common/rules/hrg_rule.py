import json
from collections import OrderedDict

from source.common.bolinas.hgraph import Hgraph


class HRGRule:
    def __init__(
        self, lhs, rhs_string, triplet, triplet_id, weight=0, predicate_info=False
    ):
        self.lhs = lhs
        self.__get_rhs_string(rhs_string)
        self.triplet_id = triplet_id
        self.predicate_info = predicate_info
        self.predicates = [f"n{p}" for p in triplet.predicate()]
        self.predicate_edges = []
        if self.predicate_info:
            self.__get_predicate_info()
        self.failed_predicate_mapping = False
        self.weight = weight

    def print_rule(self):
        return (
            f"{self.get_rule_string()}" f";{self.get_predicate_ids()}" f";{self.weight}"
        )

    def get_rule_string(self):
        return f"{self.lhs} -> {self.rhs.to_string()}"

    def get_rule_and_predicate_ids(self):
        return f"{self.lhs} -> {self.rhs.to_string()};{self.get_predicate_ids()}"

    def get_predicate_ids(self):
        return f"{'_'.join([str(p) for p in self.predicate_order_ids]) if self.predicate_info else ''}"

    def __get_rhs_string(self, rhs_string):
        rhs_from_string = Hgraph.from_string(rhs_string)
        self.rhs = Hgraph.from_string(rhs_from_string.to_bolinas_str(nodeids=True))

    def __get_predicate_info(self):
        self.node_to_order = self.rhs.get_nodes()
        self.predicate_order_ids = [self.node_to_order[p] for p in self.predicates]
        self.rhs_with_internal_ids = Hgraph.from_string(self.rhs.to_string())
        self.internal_to_order = self.rhs_with_internal_ids.get_nodes()
        self.order_to_internal = OrderedDict(
            sorted(
                {o: i for i, o in self.internal_to_order.items()}.items(),
                key=lambda x: x[0],
            ),
        )
        self.predicate_internal_ids = [
            self.order_to_internal[p] for p in self.predicate_order_ids
        ]
        for p_n, p_i in zip(self.predicates, self.predicate_internal_ids):
            edges_from_p_n = self.rhs[p_n]
            edges_from_p_i = self.rhs_with_internal_ids[p_i]
            self.predicate_edges.append(
                {f"{p_n}": edges_from_p_n, f"{p_i}": edges_from_p_i}
            )
            if edges_from_p_n.keys() != edges_from_p_i.keys():
                self.failed_predicate_mapping = True

    def log_hrg_rule(self):
        log_str = (
            f"\nRULE:\n{self.print_rule()}\n"
            f"\nRule rhs with node ids:\n{self.rhs.to_bolinas_str(nodeids=True)}"
        )
        if self.predicate_info:
            log_str += (
                f"\nRule rhs with order ids:\n{self.rhs.to_bolinas_str(orderids=True)}\n"
                f"\nRule rhs with internal ids:\n{self.rhs_with_internal_ids.to_bolinas_str(nodeids=True)}"
                f"\nRule rhs with order ids:\n{self.rhs_with_internal_ids.to_bolinas_str(orderids=True)}\n"
                f"\nPredicates: {self.predicates}"
                f"\nPredicate order IDs: {self.predicate_order_ids}"
                f"\nPredicate internal IDs: {self.predicate_internal_ids}\n"
                f"\nNode to order ID:\n{json.dumps(self.node_to_order, indent=4)}"
                f"\nOrder ID to internal ID:\n{json.dumps(self.order_to_internal, indent=4)}"
                f"\nInternal ID to order ID:\n{json.dumps(self.internal_to_order, indent=4)}\n"
                f"\nPredicate edges:\n{json.dumps(self.predicate_edges, indent=4)}"
            )
        return log_str
