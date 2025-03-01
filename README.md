# Method documentation

## Setup

`# python version = 3.10`

## Proof of concept 

First, we try to find a top estimate for our system in order to validate our concept.

### Get the data

```bash
export DATA_DIR=$HOME/data
mkdir $DATA_DIR
cd $DATA_DIR
# Download and unzip the lsoie data into a folder called lsoie_data
```

### Train a grammar

```bash
# Change to source directory
cd source
```

 We [train](source/steps/train/train.py) a hyperedge replacement [grammar](source/output/grammar) (HRG) using the [lsoie dataset](https://github.com/Jacobsolawetz/large-scale-oie/tree/master/dataset_creation/lsoie_data) on the triplet induced sub-graphs of the UD graph of a sentence. We create one rule per word and use the nonterminals `S`, `A`, `P` and `X` (no label). 

```bash
# Preprocess the train data
python steps/preproc/preproc.py  -d $DATA_DIR -c config/preproc_train.json

# Train the grammar
python steps/train/train.py -d $DATA_DIR -c config/train_per_word.json
```

#### Run the whole train pipeline

```bash
python steps/pipeline/pipeline.py -d $DATA_DIR -c config/pipeline_train.json
```

### Predict with the grammar on dev

First, we [preprocess](source/steps/preproc/preproc.py) the dev data as well.

```bash
python steps/preproc/preproc.py -d $DATA_DIR -c config/preproc_dev.json
```

Using the grammar, first we [parse](source/steps/parse/parse.py) the UD graphs on the dev set, saving the resulting charts as an intermediary output. We prune the parsing above 50.000 steps. The parsing takes 4-10 hours, see more in the [logs](source/log).

```bash
python steps/parse/parse.py -d $DATA_DIR -c config/parse_100.json
```

We [search](source/steps/kbest/kbest.py) for the top k best derivations in the chart. We apply different filters on the chart: `basic` (no filtering), `max` (searching only among the largest derivations), or classic retrieval metrics `precision`, `recall` and `f1-score` (called `pr_best` in the config), where we cheat by using the gold data and returning for each gold entry only the one derivation with the highest respective score. To calculate these scores we use the same triplet matching and scoring function as in the evaluation step. Our system returns at most k node-label maps for a sentence (triplets), where a label corresponds to a nonterminal symbol. If no predicate is present, we [resolve](source/common/triplet/triplet.py) it using the pos tags and the topological order of the nodes. Argument indices are assigned in ascending order of the node identifiers (word indices), where one argument group represents one connected subtree, where all nodes got the `A` nonterminal label during the derivation process. For `precision`, `recall` and `f1-score` we also calculate all the possible permutations of the identified argument groups. This search takes from 1 our to 2.5 days, see more in the [logs](source/log).

```bash
python steps/kbest/kbest.py -d $DATA_DIR -c config/kbest_100.json
```

### Create random predictions for comparison

We implement a [random extractor](steps/random/random_extractor.py) that uses the [artefacts](pipeline/output/artefacts) of the training dataset (distribution of the number of extractions per sentence, and distribution of labels per length of the sentence) and assures that the predicate is a verb.  

```bash
# Extract artefacts
python steps/random/artefacts.py -d $DATA_DIR -c config/artefacts_train.json

# Get random extractions
python steps/random/random_extractor.py -d $DATA_DIR -c config/random_dev.json

# Merge the extractions
python steps/predict/merge.py -d $DATA_DIR -c config/merge_dev_random.json

# Or run as a pipeline
python pipeline/pipeline.py -d $DATA_DIR -c config/pipeline_dev_random.json
```

### Evaluate the predictions

We [evaluate](source/steps/eval/eval.py) our system using a slightly modified [version](source/common/scores/sentence_scorer.py) of the [WiRe paper](https://aclanthology.org/W19-4002/) (since lsoie triplets do not necessarily have a second argument, common words are only needed for predicates (`P`) and first arguments (`A0`) in order for two triplets to match). We present the results of [all](source/output/eval/eval_dev_all.md) our systems and a filtered table for the [top estimation](source/output/eval/eval_dev_best.md).

```bash
# Eval all
python steps/eval/eval.py -d $DATA_DIR -c config/eval_dev_all.json

# Eval best
python steps/eval/eval.py -d $DATA_DIR -c config/eval_dev_best.json
```

We calculate some [statistics](steps/stat/run_all_stat.py) (distribution of extractions per sentence, predicate recognition, rule usage) for quantitative and qualitative analysis. See output [here](pipeline/output/stat).

```bash
python steps/stat/run_all_stat.py -d $DATA_DIR -c config/stat_dev.json
```

### Compare the results with baselines on the test set

We compare our [results](test/reports/eval.md) on the test set with [baseline systems](https://github.com/Jacobsolawetz/large-scale-oie/tree/master/large_scale_oie/evaluation) made available in the repository for the [lsoie paper](https://aclanthology.org/2021.eacl-main.222/).

```python
# TBD
```
