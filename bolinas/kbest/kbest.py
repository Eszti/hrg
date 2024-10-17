import json
import math
import os.path
import pickle
import time

from argparse import ArgumentParser
from collections import OrderedDict
from copy import copy
from datetime import datetime

from tuw_nlp.sem.hrg.bolinas.common.exceptions import DerivationException
from tuw_nlp.sem.hrg.bolinas.common.oie import get_rules, get_labels
from tuw_nlp.sem.hrg.bolinas.common.output import print_shifted, format_derivation
from tuw_nlp.sem.hrg.bolinas.kbest.filter.pr_filter import filter_for_pr
from tuw_nlp.sem.hrg.common.preproc import get_gold_labels
from tuw_nlp.sem.hrg.bolinas.kbest.filter.size_filter import filter_for_size
from tuw_nlp.sem.hrg.common.conll import get_pos_tags
from tuw_nlp.sem.hrg.common.io import get_range, log_to_console_and_log_lines


def get_k_best_unique_derivation(chart, k):
    kbest_unique_nodes = set()
    kbest_unique_derivations = []
    for score, derivation in chart:
        final_item = derivation[1]["START"][0]
        nodes = sorted(list(final_item.nodeset), key=lambda node: int(node[1:]))
        nodes_str = " ".join(nodes)
        if nodes_str not in kbest_unique_nodes:
            kbest_unique_nodes.add(nodes_str)
            kbest_unique_derivations.append((score, derivation))
        if len(kbest_unique_derivations) >= k:
            break
    assert len(kbest_unique_derivations) == len(kbest_unique_nodes)
    if len(kbest_unique_derivations) < k:
        print(f"Found only {len(kbest_unique_derivations)} derivations.")
    return kbest_unique_derivations


def extract_for_kth_derivation(derivation, n_score, matches_lines, rules_lines, sen_log_lines, ki):
    shifted_derivation = print_shifted(derivation)
    matches_lines.append(f"%s;%g\n" % (shifted_derivation, n_score))

    formatted_derivation = format_derivation(derivation)
    rules_lines.append("%s\t#%g\n" % (formatted_derivation, n_score))
    rules = get_rules(derivation)
    for grammar_nr, rule_str in sorted(rules.items()):
        prob = rule_str.split(';')[1].strip()
        rule = rule_str.split(';')[0].strip()
        rules_lines.append("%s\t%.2f\t%s\n" % (grammar_nr, float(prob), rule))
    rules_lines.append("\n")

    final_item = derivation[1]["START"][0]
    nodes = sorted(list(final_item.nodeset), key=lambda node: int(node[1:]))
    sen_log_lines.append("\nk%d:\t%s" % (ki, nodes))


def save_output(outputs):
    for (fn, lines) in outputs:
        if lines:
            with open(fn, "w") as f:
                f.writelines(lines)


def main(data_dir, config_json):
    start_time = time.time()
    logprob = True

    config = json.load(open(config_json))

    log_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "log",
        "kbest_" + config["model_dir"] + ".log"
    )
    log_lines = [
        f"Execution start: {str(datetime.now())}\n",
        f"Chart filters: {' '.join(sorted([f for f, c in config['filters'].items() if not c.get('ignore', False)]))}",
        "\n"]
    first = config.get("first", None)
    last = config.get("last", None)
    if first:
        log_to_console_and_log_lines(f"First: {first}\n", log_lines)
    if last:
        log_to_console_and_log_lines(f"Last: {last}\n", log_lines)

    score_disorder_collector = {}
    model_dir = os.path.join(data_dir, config["model_dir"])
    first_sen_to_proc = None
    last_sen_to_proc = None
    for sen_idx in get_range(model_dir, first, last):
        if first_sen_to_proc is None:
            first_sen_to_proc = sen_idx
        print(f"\nProcessing sen {sen_idx}\n")
        sen_dir_out = os.path.join(model_dir, str(sen_idx))

        bolinas_dir = f"{sen_dir_out}/bolinas"
        chart_file = f"{bolinas_dir}/sen{sen_idx}_chart.pickle"
        if not os.path.exists(chart_file):
            continue

        with open(chart_file, "rb") as f:
            chart = pickle.load(f)

        if "START" not in chart:
            print("No derivation found")
            continue

        gold_labels = get_gold_labels(f"{data_dir}/{config['preproc_dir']}", sen_idx)
        top_order = json.load(open(
            f"{data_dir}/{config['preproc_dir']}/{sen_idx}/preproc/pos_edge_graph_top_order.json"
        ))
        pos_tags = get_pos_tags(f"{data_dir}/{config['preproc_dir']}/{sen_idx}/preproc/parsed.conll")

        for name, c in sorted(config["filters"].items()):
            if c.get("ignore", False):
                continue
            print(f"Processing {name}")
            matches_lines = []
            labels_lines = []
            rules_lines = []
            sen_log_lines = []

            filtered_chart = copy(chart)
            sen_log_lines.append(f"Chart 'START' length: {len(filtered_chart['START'])}\n")
            if "chart_filter" in c:
                chart_filter = c["chart_filter"]
                assert chart_filter in ["basic", "max"]
                filtered_chart = filter_for_size(chart, chart_filter)
            sen_log_lines.append(f"Chart 'START' length after size filter: {len(filtered_chart['START'])}\n")

            derivations = filtered_chart.derivations("START")

            assert ("k" in c and "pr_metric" not in c) or ("k" not in c and "pr_metric" in c)

            labels_with_arg_idx = []
            if "k" in c:
                k_best_unique_derivations = get_k_best_unique_derivation(derivations, c["k"])
            elif "pr_metric" in c:
                metric = c["pr_metric"]
                assert metric in ["prec", "rec", "f1"]
                k_best_unique_derivations, labels_with_arg_idx = filter_for_pr(
                    derivations,
                    gold_labels,
                    metric,
                    pos_tags,
                    top_order,
                    config["arg_permutation"],
                )
            else:
                print("Neither 'k' nor 'pr_metric' is set")
                continue

            last_score = None
            score_disorder = {}
            for i, (score, derivation) in enumerate(k_best_unique_derivations):
                ki = i + 1
                if "k" in c:
                    n_score = score if logprob else math.exp(score)
                else:
                    n_score = score

                new_score = score
                if last_score:
                    if new_score > last_score:
                        order_str = "%d-%d" % (ki - 1, ki)
                        score_disorder[order_str] = (last_score, new_score)
                last_score = new_score

                try:
                    extract_for_kth_derivation(
                        derivation,
                        n_score,
                        matches_lines,
                        rules_lines,
                        sen_log_lines,
                        ki,
                    )
                    if "pr_metric" in c:
                        labels = labels_with_arg_idx[i]
                    else:
                        labels = get_labels(derivation)
                    labels_lines.append(f"{json.dumps(OrderedDict(sorted(labels.items(), key=lambda x: int(x[0]))))}\n")
                except DerivationException as e:
                    print("Could not construct derivation: '%s'. Skipping." % e)

            for i, val in score_disorder.items():
                sen_log_lines.append("%s: %g / %g\n" % (i, val[0], val[1]))
            score_disorder_collector[sen_idx] = (len(score_disorder.items()), len(k_best_unique_derivations))

            out_dir = os.path.join(bolinas_dir, name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_output(
                [
                    (f"{out_dir}/sen{sen_idx}_matches.graph", matches_lines),
                    (f"{out_dir}/sen{sen_idx}_predicted_labels.txt", labels_lines),
                    (f"{out_dir}/sen{sen_idx}_derivation.txt", rules_lines),
                    (f"{out_dir}/sen{sen_idx}.log", sen_log_lines),
                ]
            )
        last_sen_to_proc = sen_idx

    log_to_console_and_log_lines(f"\nFirst sentence to process: {first_sen_to_proc}", log_lines)
    log_to_console_and_log_lines(f"Last sentence to process: {last_sen_to_proc}", log_lines)

    log_to_console_and_log_lines(f"\nExecution finish: {datetime.now()}", log_lines)
    elapsed_time = time.time() - start_time
    time_str = f"Elapsed time: {round(elapsed_time / 60)} min {round(elapsed_time % 60)} sec\n"
    log_to_console_and_log_lines(time_str, log_lines)

    num_sem = len(score_disorder_collector.keys())
    log_to_console_and_log_lines(f"Number of sentences: {num_sem}", log_lines)

    sum_score_disorder = sum([val[0] for val in score_disorder_collector.values()])
    log_to_console_and_log_lines(f"Sum of score disorders: {sum_score_disorder}", log_lines)

    avg_str = f"Average score disorders: {round(sum_score_disorder / float(num_sem), 2)}\n"
    log_to_console_and_log_lines(avg_str, log_lines)
    with open(log_file, "w") as f:
        f.writelines(log_lines)


if __name__ == "__main__":
    parser = ArgumentParser(description="Bolinas is a toolkit for synchronous hyperedge replacement grammars.")
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-c", "--config", type=str)

    args = parser.parse_args()

    main(
        args.data_dir, args.config
    )
