import os
import pandas as pd
import json
import tqdm
from typing import Dict, List
from collections import defaultdict

from feilian.agents.agent import build_graph, build_state, rank_xpath_node
from feilian.etree_tools import parse_html


DATA_ROOT = "data/swde"


def program_xpath(candidates: Dict = None):
    df = pd.read_csv("data/swde_token_stats.csv")

    if not candidates:
        candidates = df[["category", "site"]].drop_duplicates().values.tolist()

    dfs = []
    for category, site in candidates:
        random_state = 0
        root_dir = "data/swde"
        query = open(f"datasets/swde/questions_cn/{category}_{site}.txt", "r").read()

        df_subset = df[(df["category"] == category) & (df["site"] == site)]
        df_subset = df_subset.sample(3, random_state=random_state)

        files = [os.path.join(root_dir, x) for x in df_subset["file_path"]]
        graph = build_graph()
        state = build_state(files, query)

        state = graph.invoke(state, config={"configurable": {"thread_id": "1"}})
        result_df = rank_xpath_node(state, category, site)
        dfs.append(result_df)
    df = pd.concat(dfs)
    return df


def run_xpath(file_path: str, xpaths: List[Dict]):
    html = open(file_path, "r").read()
    tree = parse_html(html)

    data = defaultdict(list)
    for obj in xpaths:
        try:
            results = tree.xpath(
                obj["xpath"], namespaces={"re": "http://exslt.org/regular-expressions"}
            )
        except Exception:
            print(f"Not valid xpath: {obj['xpath']}")
            results = []

        if len(results) == 0:
            continue
        results = [x.strip() for x in results if x.strip()]
        results = [x.replace("  ", " ") for x in results]
        data[obj["field_name"]] += results

    return dict(data)


def eval_objects(predict, ground_truth):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    predicted_values = set([x for v in predict.values() for x in v])
    ground_truth_values = set([x for v in ground_truth.values() for x in v])

    for value in predicted_values:
        if value in ground_truth_values:
            true_positives += 1
        else:
            false_positives += 1

    for value in ground_truth_values:
        if value not in predicted_values:
            false_negatives += 1

    return true_positives, false_positives, false_negatives


def eval(xpath_df: pd.DataFrame, candidates=None):
    data_df = pd.read_csv("data/swde_token_stats.csv")

    if not candidates:
        candidates = xpath_df[["category", "site"]].drop_duplicates().values.tolist()

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    eval_metrics = []
    for category, site in tqdm.tqdm(candidates):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        ground_truth_df = data_df[
            (data_df["category"] == category) & (data_df["site"] == site)
        ]
        ground_truth_df = ground_truth_df.sample(128, random_state=0)
        xpath_df_subset = xpath_df[
            (xpath_df["category"] == category) & (xpath_df["site"] == site)
        ]
        for i, row in ground_truth_df.iterrows():
            predict = run_xpath(
                os.path.join(DATA_ROOT, row["file_path"]),
                xpath_df_subset.to_dict(orient="records"),
            )
            ground_truth = json.loads(row["attributes"])
            tp, fp, fn = eval_objects(predict, ground_truth)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        eval_metrics.append((category, site, precision, recall, f1))

        overall_tp += true_positives
        overall_fp += false_positives
        overall_fn += false_negatives

    precision = overall_tp / (overall_tp + overall_fp + 1e-6)
    recall = overall_tp / (overall_tp + overall_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    eval_metrics.append(("overall", "overall", precision, recall, f1))

    eval_df = pd.DataFrame(
        eval_metrics, columns=["category", "site", "precision", "recall", "f1"]
    )
    return eval_df


if __name__ == "__main__":
    candidates = [("camera", "pcnation")]
    df = program_xpath(candidates)
    # df = pd.read_csv("ranked_xpaths.csv")
    df = eval(df, candidates)
    pass
