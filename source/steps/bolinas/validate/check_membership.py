from source.steps.bolinas.common.exceptions import NotAllNodesCoveredException
from source.steps.bolinas.common.hgraph.hgraph import Hgraph


def check_membership(parser, bolinas_graph, logger):
    derivation = None
    logger.log("\nVALIDATION:\n")
    input_graph = Hgraph.from_string(bolinas_graph)
    orig_nodes = sorted(list(input_graph.get_nodes().keys()), key=lambda node: int(node[1:]))
    parse_generator = parser.parse_graphs([input_graph], partial=False, logger=logger)

    for i, cky_chart in enumerate(parse_generator):
        assert i == 0
        if cky_chart.no_derivation():
            logger.log("No derivation found\n")
        else:
            derivation_list = cky_chart.search_derivations("START", only_first=True, logger=logger)
            derivation = derivation_list[0]
            derivation.full_log(logger=logger, k=1)

            not_covered_nodes = sorted(set(orig_nodes) - set(derivation.derived_nodes), key=lambda node: int(node[1:]))
            if len(not_covered_nodes) != 0:
                raise NotAllNodesCoveredException(orig_nodes, derivation.derived_nodes, not_covered_nodes)

    return derivation
