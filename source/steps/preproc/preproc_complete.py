from tuw_nlp.graph.ud_graph import UDGraph

from source.steps.preproc.preproc import Preproc


class PreprocComplete(Preproc):

    def __init__(self, config=None):
        super().__init__(
            description="Script to preprocess conll triplet data for the complete oie task.",
            script_name="preproc_complete",
            config=config,
        )

    def _do_for_triplet(
        self, sen_idx, sen_dir, sen_text, last_sen_text, parsed_doc, triplet
    ):
        ud_graph = UDGraph(parsed_doc.sentences[0])
        triplet_nodes = triplet.node_to_label.keys()
        triplet_ud = ud_graph.subgraph(
            list(triplet_nodes), handle_unconnected="shortest_path"
        )
        self._save_bolinas_graph(
            triplet_ud.pos_edge_graph(),
            f"{sen_dir}/sen{sen_idx}_triplet.graph",
            f"{sen_dir}/sen{sen_idx}_triplet_graph.dot",
            triplet=triplet,
            marked_nodes=triplet_nodes,
        )
        self._save_ud(
            triplet_ud,
            f"{sen_dir}/sen{sen_idx}_triplet_ud.dot",
            triplet=triplet,
            marked_nodes=triplet_nodes,
        )


if __name__ == "__main__":
    PreprocComplete().run()
