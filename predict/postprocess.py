def resolve_pred(pred_labels, pos_tags, top_order):
    preds = [n for n, l in pred_labels.items() if l == "P"]
    if len(preds) > 0:
        return "X"
    verbs = [n for n, t in pos_tags.items() if t == "VERB"]
    if len(verbs) == 0:
        pred_labels[str(top_order[1])] = "P"
        return "A"
    if len(verbs) == 1:
        pred_labels[verbs[0]] = "P"
        return "B"
    assert len(verbs) > 1
    first_verb_idx = None
    for v_idx in verbs:
        idx = top_order.index(int(v_idx))
        if first_verb_idx is None or idx < first_verb_idx:
            first_verb_idx = idx
    first_verb_node = str(top_order[first_verb_idx])
    pred_labels[first_verb_node] = "P"
    return "C"


def add_arg_idx(extracted_labels, length):
    prev = "O"
    idx = -1
    for i in range(1, length + 1):
        if str(i) not in extracted_labels:
            extracted_labels[str(i)] = "O"
        else:
            if extracted_labels[str(i)] == "A":
                if not prev.startswith("A"):
                    idx += 1
                extracted_labels[str(i)] = "A" + str(idx)
        prev = extracted_labels[str(i)]
