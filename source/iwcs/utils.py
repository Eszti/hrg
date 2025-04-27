import networkx as nx


class OverlappingException(Exception):
    pass


def contract_args(ud_graph, index_dict, arg_map):
    arg_heads = {}
    for name, indices in index_dict.items():
        if indices:
            if set(indices) - set(ud_graph.G):
                raise OverlappingException
            arg_graph = ud_graph.subgraph(indices, handle_unconnected="shortest_path").G
            arg_nodes = set(arg_graph)
            head = list(nx.topological_sort(arg_graph))[0]
            arg_heads[name] = head
            to_remove = [n for n in arg_nodes if n != head]
            out_edges = ud_graph.G.edges(arg_nodes, data=True)

            nx.set_node_attributes(
                ud_graph.G,
                {head: {"name": f"{name}", "upos": f"{arg_map[name].upper()}"}},
            )
            for u, v, d in out_edges:
                if v not in arg_nodes and u not in arg_heads.values():
                    ud_graph.G.add_edge(head, v, color=d["color"])
            ud_graph.G.remove_nodes_from(to_remove)
    return arg_heads


def add_info_to_node(ud_graph, index_dict=None):
    if index_dict is None:
        index_dict = {}
    for node, data in ud_graph.G.nodes(data=True):
        new_name = str(node)
        name = data["name"]
        if name:
            new_name += f"\n{name}"
        data["name"] = new_name
    for label, indices in index_dict.items():
        for idx in indices:
            ud_graph.G.nodes[idx]["name"] += f"\n{label}"
            pass
