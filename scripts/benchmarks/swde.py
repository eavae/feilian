import os
import pandas as pd
import json
import tqdm
import html
import re
import tiktoken
from lxml import etree
from minify_html import minify
from typing import Dict, List
from collections import defaultdict

from feilian.agents.agent import (
    build_graph as build_program_xpath_graph,
    build_state,
)
from feilian.agents.fragments_detection_hint import (
    build_graph as build_fragment_detection_graph,
    run_operators,
)
from feilian.etree_tools import (
    to_string,
    clean_html,
    parse_html,
    extract_text_by_xpath,
    extract_text_by_css_selector,
    normalize_text,
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


def program_xpath(
    candidates: Dict = None,
    load_extraction_from: str = None,
    save_to_dir: str = "./tmp/program_xpath",
):
    """

    Args:
        candidates (Dict, optional): _description_. Defaults to None.
        load_extraction_from (str, optional): useful when doing ablation experiment. We can load information extracted state so to skip it, save time. Defaults to None.
        save_to_dir (str, optional): _description_. Defaults to "./tmp/program_xpath".

    Returns:
        _type_: _description_
    """
    df = pd.read_csv("data/swde_token_stats.csv")

    if not candidates:
        candidates = df[["category", "site"]].drop_duplicates().values.tolist()

    # create cache dir if not exists
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir, exist_ok=True)

    if load_extraction_from and not os.path.exists(load_extraction_from):
        raise FileNotFoundError(f"file {load_extraction_from} not found, please check")

    if not load_extraction_from:
        load_extraction_from = save_to_dir

    dfs = []
    graph = build_program_xpath_graph()
    random_state = 0
    for category, site in tqdm.tqdm(candidates):
        output_file = os.path.join(save_to_dir, f"{category}_{site}.json")
        load_file = os.path.join(load_extraction_from, f"{category}_{site}.json")

        if os.path.exists(output_file):
            state = json.loads(open(output_file, "r").read())
            for field_name, field_xpath in state["xpaths"].items():
                dfs.append(
                    {
                        "category": category,
                        "site": site,
                        "field_name": field_name,
                        "xpath": field_xpath,
                    }
                )
            continue

        df_subset = df[(df["category"] == category) & (df["site"] == site)]
        df_subset = df_subset.sample(3, random_state=random_state)

        if os.path.exists(load_file):
            state = json.loads(open(load_file, "r").read())

            # add raw html
            for snippet in state["snippets"]:
                category, site, page_id = snippet["id"].split("_")
                df_row = df_subset[
                    (df_subset["category"] == category)
                    & (df_subset["site"] == site)
                    & (df_subset["page_id"] == int(page_id))
                ].iloc[0]
                snippet["raw_html"] = open(
                    os.path.join(DATA_ROOT, df_row["file_path"]), "r"
                ).read()

            # remove xpath
            state["xpaths"] = {}
        else:
            files = [os.path.join(DATA_ROOT, x) for x in df_subset["file_path"]]
            ids = [f"{category}_{site}_{i}" for i in df_subset["page_id"]]
            state = build_state(files, get_prompt(category, site), ids=ids)

        state = graph.invoke(state)

        with open(output_file, "w") as f:
            for snippet in state["snippets"]:
                del snippet["raw_html"]
            f.write(json.dumps(state, ensure_ascii=False, indent=4))

        xpaths = state["xpaths"]
        for field_name, field_xpath in xpaths.items():
            dfs.append(
                {
                    "category": category,
                    "site": site,
                    "field_name": field_name,
                    "xpath": field_xpath,
                }
            )
    df = pd.DataFrame(dfs)
    return df


def build_fragment_detection_state(file_path: str, query: str, page_id: str):
    raw_html = open(file_path, "r").read()
    return {
        "id": page_id,
        "raw_html": raw_html,
        "query": query,
    }


def get_full_text(tree, target_text):
    if isinstance(tree, etree._ElementTree):
        tree = tree.getroot()

    objs = []
    for text in tree.itertext():
        normalized_text = html.unescape(html.unescape(text)).strip()
        normalized_text = re.sub(r"  +", " ", normalized_text)
        if target_text in normalized_text:
            additional_length = len(text) - len(target_text)
            objs.append(
                {
                    "full_text": text,
                    "score": 1 - additional_length / len(text),
                }
            )

    objs = sorted(objs, key=lambda x: x["score"], reverse=True)
    if objs:
        return objs[0]["full_text"]
    return target_text


def detect_fragments(candidates: Dict = None):
    df = pd.read_csv("data/swde_token_stats.csv")

    if not candidates:
        candidates = df[["category", "site"]].drop_duplicates().values.tolist()

    true_positive = 0
    false_positive = 0
    false_negative = 0

    graph = build_fragment_detection_graph()
    for category, site in candidates:
        random_state = 0
        query = get_prompt(category, site)

        df_subset = df[(df["category"] == category) & (df["site"] == site)]
        df_subset = df_subset.sample(3, random_state=random_state)

        for i, row in df_subset.iterrows():
            state = build_fragment_detection_state(
                os.path.join(DATA_ROOT, row["file_path"]),
                query,
                f"{category}_{site}_{row['page_id']}",
            )

            file_path = f"tests/data/hint_fragments/{state['id']}.json"
            if not os.path.exists(file_path):
                state = graph.invoke(state)
                with open(file_path, "w") as f:
                    del state["raw_html"]
                    del state["ops"]
                    f.write(json.dumps(state, ensure_ascii=False, indent=4))
            else:
                state = json.loads(open(file_path, "r").read())

            # tree = run_operators(tree, state["ops"])
            # html = to_string(tree)
            # minified_html = minify(html, keep_closing_tags=True)
            # tokens_after = tokenizer(minified_html)

            ground_truths = json.loads(row["attributes"])
            prediction = {}
            for field_name, ops in state["field_operators"].items():
                # prediction[field_name] = ops[-1]["data"]["value"]
                raw_html = open(os.path.join(DATA_ROOT, row["file_path"]), "r").read()
                tree = parse_html(raw_html)
                clean_html(tree)
                run_operators(tree, ops)
                values = []
                for v in ops[-1]["data"]["value"]:
                    values.append(get_full_text(tree, v))
                prediction[field_name] = values

            tp, fp, fn = eval_objects(prediction, ground_truths)
            true_positive += tp
            false_positive += fp
            false_negative += fn

    accuracy = true_positive / (true_positive + false_positive + false_negative + 1e-6)
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")


def test_fragments():
    source_df = pd.read_csv("data/swde_token_stats.csv")
    df = pd.read_csv("data/swde_fragments.csv")

    for i, row in df.iterrows():
        source_row = source_df[
            (source_df["page_id"] == row["id"])
            & (source_df["category"] == row["category"])
            & (source_df["site"] == row["site"])
        ].iloc[0]
        file_path = os.path.join(DATA_ROOT, source_row["file_path"])
        raw_html = open(file_path, "r").read()
        tree = parse_html(raw_html)
        ops = json.loads(row["ops"])
        tree = run_operators(tree, ops)
        html = minify(to_string(tree), keep_closing_tags=True)

        ground_truths = json.loads(row["ground_truth"])
        for key, values in ground_truths.items():
            if any([x in html for x in values]):
                continue

            values = [normalize_text(x) for x in values]
            if any([x in html for x in values]):
                continue
            pass


def run_xpath(file_path: str, xpaths: List[Dict]):
    raw_html = open(file_path, "r").read()
    tree = parse_html(minify(raw_html, keep_closing_tags=True))

    extract_fn = (
        extract_text_by_xpath
        if PROGRAM_TYPE == "xpath"
        else extract_text_by_css_selector
    )
    data = defaultdict(list)
    for obj in xpaths:
        data[obj["field_name"]] += extract_fn(tree, obj["xpath"])[0]

    return dict(data)


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
        predict_values = [normalize_text(x) for x in predict_values]
        ground_truth_values = [normalize_text(x) for x in ground_truth_values]
        tp, fp, fn = eval_array(predict_values, ground_truth_values)
        true_positives += tp
        false_positives += fp
        false_negatives += fn

    return true_positives, false_positives, false_negatives


def eval(xpath_df: pd.DataFrame, candidates=None, sample_size=32):
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
        ground_truth_df = ground_truth_df.sample(sample_size, random_state=0)
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
    xpath_df = program_xpath(
        load_extraction_from="tmp/program_xpath",
        save_to_dir="tmp/program_xpath_wo_q",
    )
    df = eval(xpath_df)
    pass
