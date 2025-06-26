from itertools import product

import networkx as nx


class OverlappingException(Exception):
    pass


def contract_triplet_elements(ud_graph, index_dict, element_map):
    heads = {}
    for name, indices in index_dict.items():
        if indices:
            if set(indices) - set(ud_graph.G):
                raise OverlappingException
            try:
                arg_graph = ud_graph.subgraph(
                    indices, handle_unconnected="shortest_path"
                ).G
            except nx.NetworkXNoPath:
                raise OverlappingException
            arg_nodes = set(arg_graph)
            head = list(nx.topological_sort(arg_graph))[0]
            heads[name] = head
            to_remove = [n for n in arg_nodes if n != head]
            out_edges = ud_graph.G.edges(arg_nodes, data=True)

            nx.set_node_attributes(
                ud_graph.G,
                {head: {"name": f"{name}", "upos": f"{element_map[name].upper()}"}},
            )
            for u, v, d in out_edges:
                if v not in arg_nodes and u not in heads.values():
                    ud_graph.G.add_edge(head, v, color=d["color"])
            ud_graph.G.remove_nodes_from(to_remove)
    return heads


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


def words_to_idx(arg, parsed_sen):
    tokens = [w.text for w in parsed_sen.words]
    words = arg.split(" ")
    len_arg = len(words)
    arg_idx = []
    for i in range(len(tokens) - len_arg):
        if " ".join(tokens[i : i + len_arg]) == arg:
            arg_idx.append([j for j in range(i, i + len_arg)])
    if len(arg_idx) == 0:
        all_matches = []
        for word in words:
            new_idx = [j for j, t in enumerate(tokens) if t == word]
            all_matches.append(new_idx)

        # Roll span forwards
        for k in range(1, len(all_matches)):
            if len(all_matches[k]) > 1 and len(all_matches[k - 1]) == 1:
                if all_matches[k - 1][0] + 1 in all_matches[k]:
                    all_matches[k] = [all_matches[k - 1][0] + 1]
        # Roll span backwards
        for k in range(1, len(all_matches)):
            if len(all_matches[-k]) == 1 and len(all_matches[-k - 1]) > 1:
                if all_matches[-k][0] - 1 in all_matches[-k - 1]:
                    all_matches[-k - 1] = [all_matches[-k][0] - 1]

        # Find span in multi-option ranges in between
        for k in range(len(all_matches)):
            if len(all_matches[k]) > 1:
                l = k + 1
                while l < len(all_matches) and len(all_matches[l]) > 1:
                    l += 1
                if l > k + 1:
                    for idx in all_matches[k]:
                        next_idx = idx + 1
                        found = True
                        for m in range(k + 1, l):
                            if next_idx in all_matches[m]:
                                next_idx += 1
                            else:
                                found = False
                                break
                        if found:
                            new_idx = idx
                            for m in range(k, l):
                                assert new_idx in all_matches[m]
                                all_matches[m] = [new_idx]
                                new_idx += 1
                            break

        # Remove uniques from multi-options
        uniques = set()
        multiples = []
        for k in range(len(all_matches)):
            if len(all_matches[k]) == 1:
                uniques.add(all_matches[k][0])
            elif len(all_matches[k]) > 1:
                multiples.append(k)
        for k in multiples:
            new_idx_l = list(set(all_matches[k]) - uniques)
            if len(new_idx_l) == 0:
                return []
            all_matches[k] = new_idx_l
            if len(new_idx_l) == 1:
                uniques.add(new_idx_l[0])

        # Take closest in single multi-options - first
        if len(all_matches[0]) > 1 and len(all_matches[1]) == 1:
            sorted_idx = sorted(all_matches[0], reverse=True)
            for idx in sorted_idx:
                if idx < all_matches[1][0]:
                    all_matches[0] = [idx]
                    break
        # Take closest in single multi-options - last
        if len(all_matches[-1]) > 1 and len(all_matches[-2]) == 1:
            sorted_idx = sorted(all_matches[-1])
            for idx in sorted_idx:
                if idx > all_matches[-2][0]:
                    all_matches[-1] = [idx]
                    break
        # Take closest in single multi-options - middle
        for k in range(1, len(all_matches) - 1):
            if (
                len(all_matches[k]) > 1
                and len(all_matches[k - 1]) == 1
                and len(all_matches[k + 1]) == 1
            ):
                for idx in all_matches[k]:
                    if all_matches[k - 1][0] < idx < all_matches[k + 1][0]:
                        all_matches[k] = [idx]

        # Take one-one for duplicates after each other
        for k in range(len(all_matches) - 1):
            if len(all_matches[k]) > 1 and sorted(all_matches[k]) == sorted(
                all_matches[k + 1]
            ):
                sorted_duplicates = sorted(all_matches[k])
                all_matches[k] = [sorted_duplicates[0]]
                all_matches[k + 1] = [sorted_duplicates[1]]

        # Backprop from first single
        first_single = 0
        for k in range(len(all_matches)):
            if len(all_matches[k]) == 1:
                first_single = k
                break
        if first_single > 0:
            highest_idx = all_matches[first_single][0]
            for k in sorted(range(first_single), reverse=True):
                for candidate in sorted(all_matches[k], reverse=True):
                    if candidate < highest_idx:
                        all_matches[k] = [candidate]
                        highest_idx = candidate
                        break

        arg_idx = list(product(*all_matches))
    # Check unique indices
    if len(arg_idx) > 1:
        arg_idx = [arg_idx[0]]
    if len(arg_idx) == 0:
        return []

    assert len(arg_idx) == 1
    arg_idx = [idx + 1 for idx in arg_idx[0]]
    return arg_idx
