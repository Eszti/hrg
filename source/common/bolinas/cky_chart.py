import heapq
import itertools
import pickle
import time
from collections import Counter
from copy import copy

from source.common.derivation.derivation_list import DerivationList


class CkyChart:
    def __init__(self, items_dict):
        self.chart = items_dict

    @staticmethod
    def from_pickle(fn):
        with open(fn, "rb") as f:
            chart = pickle.load(f)
        return CkyChart(chart)

    def to_file(self, fn):
        with open(fn, "wb") as f:
            pickle.dump(self.chart, f, -1)

    def no_derivation(self):
        return "START" not in self.chart

    def chart_with_only_max_size(self):
        boundary_value = sorted(self._get_sizes().keys(), reverse=True)[0]
        filtered_chart = self._delete_smaller(boundary_value)
        return CkyChart(filtered_chart)

    def _get_sizes(self):
        graph_sizes = Counter()
        for split in self.chart["START"]:
            assert len(split.items()) == 1
            graph_size = len(split["START"].nodeset)
            graph_sizes[graph_size] += 1
        return graph_sizes

    def _delete_smaller(self, boundary_value):
        new_chart = copy(self.chart)
        splits_to_keep = []
        for split in self.chart["START"]:
            assert len(split.items()) == 1
            graph_size = len(split["START"].nodeset)
            if graph_size >= boundary_value:
                splits_to_keep.append(split)
        del new_chart["START"]
        new_chart["START"] = splits_to_keep
        return new_chart

    def search_derivations(
        self,
        item="START",
        only_first=False,
        max_steps=None,
        k_best=None,
        sen_logger=None,
        global_logger=None,
    ):
        start_time = time.time()
        if only_first:
            derivation, steps = self._first_derivation(item)
            derivations = [derivation]
        else:
            derivations, steps = self._derivations(item, 0, max_steps, k_best)
        elapsed_time = round(time.time() - start_time, 2)
        search_time = f"Search time: {elapsed_time} sec"
        search_steps = f"Search steps: {steps}"
        nr_derivations = f"Number of derivations: {len(derivations)}"
        print(
            f"Search: {elapsed_time} sec, {steps} steps, {len(derivations)} derivation(s)"
        )
        if sen_logger:
            sen_logger.log(f"\n{search_time}")
            sen_logger.log(f"{search_steps}")
            sen_logger.log(f"{nr_derivations}\n")
        if global_logger:
            global_logger.log(f"{search_steps}")
            global_logger.log(f"{nr_derivations}\n")
        return DerivationList(derivations)

    def _derivations(self, item, done_steps, max_steps, k_best):
        """
        Return all derivations from this chart.
        """

        if item == "START":
            rprob = 0.0
        else:
            rprob = item.rule.weight

        # If item is a leaf, just return it and its probability
        if not item in self.chart:
            if item == "START":
                print("No derivations.")
                return [], 1
            else:
                return [(rprob, item)], 1

        pool = []
        all_steps = done_steps
        splits = self.chart[item]
        for split in splits:
            if max_steps is None or all_steps < max_steps:
                nts, children = zip(*split.items())
                children_derivations = [
                    self._derivations(child, all_steps, max_steps, k_best)
                    for child in children
                ]
                kbest_each_child, steps_each_child = zip(*children_derivations)
                all_steps += sum(steps_each_child)

                all_combinations = list(itertools.product(*kbest_each_child))

                combinations_for_sorting = []
                for combination in all_combinations:
                    weights, trees = zip(*combination)
                    try:
                        heapq.heappush(
                            combinations_for_sorting, (sum(weights) + rprob, trees)
                        )
                    except TypeError:
                        pass

                    for prob, trees in sorted(
                        combinations_for_sorting, key=lambda x: x[0], reverse=True
                    )[:k_best]:
                        new_tree = (item, dict(zip(nts, trees)))
                        try:
                            heapq.heappush(pool, (prob, new_tree))
                        except TypeError:
                            pass

        return (
            sorted(pool, key=lambda x: x[0], reverse=True)[:k_best],
            all_steps - done_steps + 1,
        )

    def _first_derivation(self, item):
        """
        Return one derivation from this chart.
        """
        if item == "START":
            rprob = 0.0
        else:
            rprob = item.rule.weight

        if not item in self.chart:
            if item == "START":
                print("No derivations.")
                return None
            else:
                return (rprob, item), 1

        split = list(self.chart[item])[0]
        nts, children = zip(*split.items())
        children_derivations = [self._first_derivation(child) for child in children]
        one_from_each_child, steps = zip(*children_derivations)

        weights, trees = zip(*one_from_each_child)
        new_tree = (item, dict(zip(nts, trees)))
        new_prob = sum(weights) + rprob
        return (new_prob, new_tree), sum(steps) + 1

    def items_length(self):
        length = 0
        splits = self.chart.items()
        for split in splits:
            for item_dict in split[1]:
                length += len(item_dict.values())
        return length

    def log_items_len(self):
        return f"Chart items len: {self.items_length()}"

    def log_length(self):
        return (
            f"Chart START items len: {len(self.chart['START']) if 'START' in self.chart else 0}\n"
            f"Chart keys len: {len(self.chart)}\n"
            f"Chart items len: {self.items_length()}"
        )
