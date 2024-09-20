import os
import pandas as pd
import json
import tqdm
import re
import html
from typing import Dict, List
from collections import defaultdict

from feilian.agents.agent import build_graph, build_state, rank_xpath_node
from feilian.etree_tools import parse_html, to_string
from feilian.text_tools import convert_html_to_text


DATA_ROOT = "data/swde"


def program_xpath(candidates: Dict = None):
    df = pd.read_csv("data/swde_token_stats.csv")

    if not candidates:
        candidates = df[["category", "site"]].drop_duplicates().values.tolist()

    dfs = []
    graph = build_graph()
    for category, site in candidates:
        random_state = 0
        root_dir = "data/swde"
        query = open(f"datasets/swde/questions_cn/{category}_{site}.txt", "r").read()

        df_subset = df[(df["category"] == category) & (df["site"] == site)]
        df_subset = df_subset.sample(3, random_state=random_state)

        files = [os.path.join(root_dir, x) for x in df_subset["file_path"]]
        ids = [f"{category}_{site}_{i}" for i in df_subset["page_id"]]
        state = build_state(files, query, ids=ids)

        state = graph.invoke(state, config={"configurable": {"thread_id": "1"}})
        result_df = rank_xpath_node(state, category, site)
        dfs.append(result_df)
    df = pd.concat(dfs)
    return df


def run_xpath(file_path: str, xpaths: List[Dict]):
    raw_html = open(file_path, "r").read()
    tree = parse_html(raw_html)

    data = defaultdict(list)
    for obj in xpaths:
        try:
            # replace all single quotes with double quotes
            obj["xpath"] = obj["xpath"].replace("'", '"')
            # replace \" with '
            obj["xpath"] = obj["xpath"].replace('\\"', "'")

            results = []
            for ele in tree.xpath(
                obj["xpath"], namespaces={"re": "http://exslt.org/regular-expressions"}
            ):
                if not ele:
                    continue
                if isinstance(ele, str):
                    results.append(ele)
                else:
                    results.append(convert_html_to_text(to_string(ele)))

        except Exception:
            print(f"Not valid xpath: {obj['xpath']}")
            results = []

        if len(results) == 0:
            continue
        results = [html.unescape(x) for x in results]
        results = [x.strip() for x in results if x.strip()]
        results = [re.sub(r"\s+", " ", x) for x in results]
        data[obj["field_name"]] += results

    return dict(data)


def unescape_and_strip(text):
    return html.unescape(text).strip()


def eval_array(predict: List[str], ground_truth: List[str]):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for value in predict:
        if value in ground_truth:
            true_positives += 1
        else:
            false_positives += 1

    for value in ground_truth:
        if value not in predict:
            false_negatives += 1

    return true_positives, false_positives, false_negatives


def eval_objects(predict, ground_truth):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for field_name, predict_values in predict.items():
        ground_truth_values = ground_truth.get(field_name, [])
        predict_values = [unescape_and_strip(x) for x in predict_values]
        ground_truth_values = [unescape_and_strip(x) for x in ground_truth_values]
        tp, fp, fn = eval_array(predict_values, ground_truth_values)
        true_positives += tp
        false_positives += fp
        false_negatives += fn

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

        accuracy = true_positives / (
            true_positives + false_positives + false_negatives + 1e-6
        )
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        eval_metrics.append((category, site, accuracy, precision, recall, f1))

        overall_tp += true_positives
        overall_fp += false_positives
        overall_fn += false_negatives

    accuracy = overall_tp / (overall_tp + overall_fp + overall_fn + 1e-6)
    precision = overall_tp / (overall_tp + overall_fp + 1e-6)
    recall = overall_tp / (overall_tp + overall_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    eval_metrics.append(("overall", "overall", accuracy, precision, recall, f1))

    eval_df = pd.DataFrame(
        eval_metrics,
        columns=["category", "site", "accuracy", "precision", "recall", "f1"],
    )
    return eval_df


if __name__ == "__main__":
    # candidates = [("university", "ecampustours")]
    xpath_df = program_xpath()
    xpath_df.to_csv("data/swde_xpath_program.csv", index=False)
    # df = pd.read_csv("ranked_xpaths.csv")
    eval_df = eval(xpath_df)
    eval_df.to_csv("data/swde_xpath_program_eval.csv", index=False)
    pass
