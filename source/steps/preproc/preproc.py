import json
from abc import abstractmethod
from collections import defaultdict

import networkx as nx
import stanza
from stanza.utils.conll import CoNLL
from tuw_nlp.graph.ud_graph import UDGraph

from source.common.script.loop_on_conll import LoopOnConll
from source.common.triplet.triplet import Triplet
from source.common.triplet.triplets_for_sen import TripletsForSen


class Preproc(LoopOnConll):

    def __init__(self, description, script_name, log=True, config=None):
        super().__init__(description, log=log, script_name=script_name, config=config)
        self.gold_sen_text, self.gold_sen_id, self.gold_triplets = None, None, []

    def _before_loop(self):
        self.nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,mwt,pos,lemma,depparse",
            tokenize_pretokenized=True,
        )

    def _do_for_sen(self, sen_idx, sen, sen_txt, last_sen_txt, sen_dir):
        self._save_conll(sen, f"{sen_dir}/sen{sen_idx}.conll")
        parsed_doc = self._parse_doc(
            sen, sen_dir, save=(sen_txt != last_sen_txt) or self.folder_per_sen
        )

        if sen_txt != last_sen_txt or self.folder_per_sen:
            ud_graph = UDGraph(parsed_doc.sentences[0])
            json.dump(
                [n for n in nx.topological_sort(ud_graph.G)],
                open(f"{sen_dir}/graph_top_order.json", "w"),
            )
            self._save_bolinas_graph(
                ud_graph.pos_edge_graph(),
                f"{sen_dir}/pos_edge.graph",
                f"{sen_dir}/pos_edge_graph.dot",
            )
            self._save_ud(ud_graph, f"{sen_dir}/general_ud.graph")
            if self.gold_sen_id is not None:
                self._save_gold_triplets()
                self.gold_sen_text, self.gold_sen_id, self.gold_triplets = (
                    None,
                    None,
                    [],
                )

        triplet = self.__get_triplet(sen, sen_idx)
        if self.gold_sen_id is None:
            self.gold_sen_id = sen_idx
            self.gold_sen_text = sen_txt
        self.gold_triplets.append(triplet)
        self._do_for_triplet(
            sen_idx, sen_dir, sen_txt, last_sen_txt, parsed_doc, triplet
        )

    @abstractmethod
    def _do_for_triplet(
        self, sen_idx, sen_dir, sen_text, last_sen_text, parsed_doc, triplet
    ):
        raise NotImplemented

    def _save_bolinas_graph(
        self, pos_edge_graph, graph_fn, dot_fn, triplet=None, marked_nodes=None
    ):
        self.__save_bolinas_str(graph_fn, pos_edge_graph)
        if triplet is not None:
            self.__add_triplet_labels_to_name(pos_edge_graph, triplet)
        self.__add_node_id_to_name(pos_edge_graph)
        self.save_as_dot(dot_fn, pos_edge_graph, marked_nodes=marked_nodes)

    def _save_ud(self, ud_graph, dot_fn, triplet=None, marked_nodes=None):
        if triplet is not None:
            self.__add_triplet_labels_to_name(ud_graph, triplet)
        self.__add_node_id_to_name(ud_graph)
        self.save_as_dot(dot_fn, ud_graph, marked_nodes=marked_nodes)

    @staticmethod
    def __get_triplet(sen, sen_id):
        triplet_dict = defaultdict(list)
        for i, tok in enumerate(sen):
            label = tok[7].split("-")[0]
            if label == "P" or label.startswith("A"):
                triplet_dict[label].append(i + 1)
        return Triplet(triplet_dict, triplet_id=sen_id)

    @staticmethod
    def __add_triplet_labels_to_name(graph, triplet, node_prefix=""):
        for n in graph.G.nodes:
            key = n
            if node_prefix:
                key = n.split(node_prefix)[1]
            label = triplet.get_label(key)
            if label is None:
                label = ""
            new_name = graph.G.nodes[n]["name"]
            if new_name:
                new_name += "\n"
            new_name += f"{label}"
            graph.G.nodes[n]["name"] = new_name

    @staticmethod
    def __add_node_id_to_name(ud_graph):
        for node, data in ud_graph.G.nodes(data=True):
            new_name = str(node)
            name = data["name"]
            if name:
                new_name += f"\n{name}"
            data["name"] = new_name

    @staticmethod
    def _save_conll(sen, fn):
        with open(fn, "w") as f:
            for line in sen:
                line[0] = str(int(line[0]) + 1)
                f.write("\t".join(line))
                f.write("\n")

    def _parse_doc(self, sen, out_dir, save=True):
        parsed_doc = self.nlp(" ".join(t[1] for t in sen))
        if save:
            fn = f"{out_dir}/parsed.conll"
            CoNLL.write_doc2conll(parsed_doc, fn)
        return parsed_doc

    @staticmethod
    def __save_bolinas_str(fn, graph, add_names=False):
        bolinas_graph = graph.to_bolinas(add_names=add_names)
        with open(fn, "w") as f:
            f.write(f"{bolinas_graph}\n")

    @staticmethod
    def save_as_dot(fn, graph, marked_nodes=None):
        if marked_nodes is None:
            marked_nodes = set()
        with open(fn, "w") as f:
            f.write(graph.to_dot(marked_nodes))

    def _save_gold_triplets(
        self, sen_id=None, sen_text=None, triplets=None, fn_prefix="gold_triplets"
    ):
        gold_triplets = self.gold_triplets if triplets is None else triplets
        gold_sen_id = self.gold_sen_id if sen_id is None else sen_id
        gold_sen_text = self.gold_sen_text if sen_text is None else sen_text
        sen_dir = f"{self.out_dir}/{gold_sen_id}"
        triplets_for_sen = TripletsForSen(gold_triplets, gold_sen_id, gold_sen_text)
        triplets_for_sen.save_summary(f"{sen_dir}/{fn_prefix}_summary.txt")
        triplets_for_sen.to_json(f"{sen_dir}/{fn_prefix}.json")

    def _after_loop(self):
        self._save_gold_triplets()
        super()._after_loop()
