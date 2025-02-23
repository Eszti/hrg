import copy
from collections import defaultdict

import networkx as nx
from tuw_nlp.graph.ud_graph import UDGraph

from source.common.triplet.triplet import Triplet
from source.common.triplet.triplets_for_sen import TripletsForSen
from source.steps.preproc.preproc import Preproc


class PreprocContracted(Preproc):

    def __init__(self, config=None):
        super().__init__(
            description="Script to preprocess conll triplet data for the triplet extraction subtask.",
            script_name="preproc_contracted",
            config=config,
        )
        self.unconnected_arg = defaultdict(set)
        self.out_edge_from_arg = defaultdict(set)
        self.contract_triplet = self.config.get("contract_triplet", False)

    def _do_for_triplet(self, sen_idx, sen_dir, sen_text, parsed_doc, triplet):
        triplet_nodes = set(triplet.node_to_label.keys())
        self._save_ud(
            UDGraph(parsed_doc.sentences[0]),
            f"{sen_dir}/sen{sen_idx}_ud.dot",
            triplet=triplet,
            marked_nodes=triplet_nodes,
        )

        ud_graph = UDGraph(parsed_doc.sentences[0])
        arg_graphs = self.__get_argument_graphs(sen_idx, ud_graph, triplet)
        if sen_idx in self.unconnected_arg:
            self.logger.log(
                f"Unconnected arg: {sen_idx}:{self.unconnected_arg[sen_idx]}",
                to_stdout=True,
            )
            return

        arg_heads = self.__contract_args(sen_idx, ud_graph, arg_graphs, triplet)
        contracted_label_to_nodes = copy.copy(triplet.label_to_nodes)
        for l, nodes in contracted_label_to_nodes.items():
            if l.startswith("A"):
                kept_nodes = [n for n in nodes if n in arg_heads]
                assert len(kept_nodes) == 1
                contracted_label_to_nodes[l] = kept_nodes
        contracted_triplet = Triplet(contracted_label_to_nodes, triplet_id=sen_idx)
        contracted_triplets_for_sen = TripletsForSen(
            [contracted_triplet], sen_idx, sen_text
        )
        contracted_triplets_for_sen.save_summary(
            f"{sen_dir}/sen{sen_idx}_gold_contracted_triplets_summary.txt"
        )
        contracted_triplets_for_sen.to_json(
            f"{sen_dir}/sen{sen_idx}_gold_contracted_triplets.json"
        )

        contracted_triplet_nodes = arg_heads + triplet.predicate()
        self._save_bolinas_graph(
            ud_graph.pos_edge_graph(),
            f"{sen_dir}/sen{sen_idx}_contracted.graph",
            f"{sen_dir}/sen{sen_idx}_contracted_graph.dot",
            triplet=triplet,
            marked_nodes=contracted_triplet_nodes,
        )

        triplet_ud = ud_graph.subgraph(
            contracted_triplet_nodes, handle_unconnected="shortest_path"
        )
        self._save_bolinas_graph(
            triplet_ud.pos_edge_graph(),
            f"{sen_dir}/sen{sen_idx}_triplet.graph",
            f"{sen_dir}/sen{sen_idx}_triplet_graph.dot",
            triplet=triplet,
            marked_nodes=contracted_triplet_nodes,
        )

        self._save_ud(
            ud_graph,
            f"{sen_dir}/sen{sen_idx}_contracted_ud.dot",
            triplet=triplet,
            marked_nodes=contracted_triplet_nodes,
        )
        self._save_ud(
            triplet_ud,
            f"{sen_dir}/sen{sen_idx}_triplet_ud.dot",
            triplet=triplet,
            marked_nodes=contracted_triplet_nodes,
        )

    def __contract_args(self, sen_idx, ud_graph, arg_graphs, triplet):
        arg_heads = []
        for arg, arg_graph in arg_graphs.items():
            arg_nodes = triplet.arguments()[arg]
            assert set(arg_nodes) == set(arg_graph)
            head = list(nx.topological_sort(arg_graph))[0]
            arg_heads.append(head)
            to_remove = [n for n in arg_nodes if n != head]
            out_edges = ud_graph.G.edges(arg_nodes, data=True)

            nx.set_node_attributes(
                ud_graph.G,
                {head: {"name": f"arg{arg.split('A')[-1]}", "upos": arg.upper()}},
            )
            for u, v, d in out_edges:
                if v not in arg_nodes and u not in arg_heads:
                    self.out_edge_from_arg[sen_idx].add(arg)
                    ud_graph.G.add_edge(head, v, color=d["color"])
            ud_graph.G.remove_nodes_from(to_remove)
        return arg_heads

    def __get_argument_graphs(self, sen_id, ud_graph, triplet):
        a_graphs = {}
        for arg, nodes in triplet.arguments().items():
            for n in nodes:
                assert n in ud_graph.G.nodes()
            a_graph = ud_graph.G.subgraph(nodes)
            if nx.is_weakly_connected(a_graph):
                a_graphs[arg] = a_graph
            else:
                self.unconnected_arg[sen_id].add(arg)
                a_graphs[arg] = None
        return a_graphs

    def _after_loop(self):
        self.logger.log(
            f"\nNumber of unconnected arguments: {len(self.unconnected_arg)}"
            f"\nNumber of out edges from argument: {len(self.out_edge_from_arg)}",
            to_stdout=True,
        )
        super()._after_loop()


if __name__ == "__main__":
    PreprocContracted().run()
