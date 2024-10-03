import os
import pandas as pd
import json
import tqdm
import html
import tiktoken
from minify_html import minify
from typing import Dict, List
from collections import defaultdict

from feilian.agents.agent import (
    build_graph as build_program_xpath_graph,
    build_state,
    rank_xpath_node,
)
from feilian.agents.fragments_detection import (
    build_graph as build_fragment_detection_graph,
    run_operators,
)
from feilian.etree_tools import (
    to_string,
    clean_html,
    parse_html,
    extract_text_by_xpath,
    extract_text_by_css_selector,
)


DATA_ROOT = "data/swde"
PROGRAM_TYPE = "xpath"

encoder = tiktoken.encoding_for_model("gpt-4")


def tokenizer(text):
    if not text:
        return 0
    return len(encoder.encode(text))


def get_prompt(category: str, site: str):
    return open(
        f"datasets/swde/questions_{os.getenv('PROMPT_LANG', 'cn')}/{category}_{site}.txt",
        "r",
    ).read()


def program_xpath(candidates: Dict = None):
    df = pd.read_csv("data/swde_token_stats.csv")

    if not candidates:
        candidates = df[["category", "site"]].drop_duplicates().values.tolist()

    dfs = []
    graph = build_program_xpath_graph()
    for category, site in candidates:
        random_state = 0
        root_dir = "data/swde"
        query = get_prompt(category, site)

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


def build_fragment_detection_state(file_path: str, query: str, page_id: str):
    raw_html = open(file_path, "r").read()
    return {
        "id": page_id,
        "raw_html": raw_html,
        "query": query,
    }


def detect_fragments(candidates: Dict = None):
    df = pd.read_csv("data/swde_token_stats.csv")

    if not candidates:
        candidates = df[["category", "site"]].drop_duplicates().values.tolist()

    dfs = []
    graph = build_fragment_detection_graph()
    for category, site in candidates:
        random_state = 0
        query = get_prompt(category, site)

        df_subset = df[(df["category"] == category) & (df["site"] == site)]
        df_subset = df_subset.sample(3, random_state=random_state)

        rows = []
        for i, row in df_subset.iterrows():
            state = build_fragment_detection_state(
                os.path.join(DATA_ROOT, row["file_path"]),
                query,
                row["page_id"],
            )

            tree = parse_html(state["raw_html"])
            clean_html(tree)
            tokens_before = tokenizer(minify(to_string(tree), keep_closing_tags=True))

            state = graph.invoke(state)

            tree = run_operators(tree, state["ops"])
            html = to_string(tree)
            minified_html = minify(html, keep_closing_tags=True)
            tokens_after = tokenizer(minified_html)

            tp, fp, fn = eval_objects(state["extracted"], json.loads(row["attributes"]))
            accuracy = tp / (tp + fp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            rows.append(
                {
                    "id": state["id"],
                    "category": category,
                    "site": site,
                    "ops": json.dumps(state["ops"]),
                    "prediction": json.dumps(state["extracted"]),
                    "ground_truth": row["attributes"],
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        dfs.append(pd.DataFrame(rows))
    df = pd.concat(dfs)
    return df


def run_xpath(file_path: str, xpaths: List[Dict]):
    raw_html = open(file_path, "r").read()
    tree = parse_html(raw_html)

    extract_fn = (
        extract_text_by_xpath
        if PROGRAM_TYPE == "xpath"
        else extract_text_by_css_selector
    )
    data = defaultdict(list)
    for obj in xpaths:
        data[obj["field_name"]] += extract_fn(tree, obj["xpath"])

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

    for field_name in set(predict.keys()) | set(ground_truth.keys()):
        ground_truth_values = ground_truth.get(field_name, [])
        predict_values = predict.get(field_name, [])
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
    predictions = []
    ground_truths = []
    for category, site in tqdm.tqdm(candidates):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        ground_truth_df = data_df[
            (data_df["category"] == category) & (data_df["site"] == site)
        ]
        ground_truth_df = ground_truth_df.sample(100, random_state=0)
        xpath_df_subset = xpath_df[
            (xpath_df["category"] == category) & (xpath_df["site"] == site)
        ]
        for i, row in ground_truth_df.iterrows():
            prediction = run_xpath(
                os.path.join(DATA_ROOT, row["file_path"]),
                xpath_df_subset.to_dict(orient="records"),
            )
            ground_truth = json.loads(row["attributes"])
            tp, fp, fn = eval_objects(prediction, ground_truth)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

            predictions.append(json.dumps(prediction))
            ground_truths.append(json.dumps(ground_truth))

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
    predictions_df = pd.DataFrame(
        {"prediction": predictions, "ground_truth": ground_truths}
    )
    return eval_df, predictions_df


if __name__ == "__main__":
    df = detect_fragments(
        candidates=[
            # ("auto", "aol"),
            # ("auto", "autobytel"),
            # ("auto", "automotive"),
            ("auto", "autoweb"),
            # ("auto", "carquotes"),
            # ("auto", "cars"),
            # ("auto", "kbb"),
            # ("auto", "motortrend"),
            # ("auto", "msn"),
            # ("auto", "yahoo"),
        ]
    )
    df.to_csv("swde_fragments.csv", index=False)
    pass
