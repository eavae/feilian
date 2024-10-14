import hashlib
import json
import pandas as pd
import warnings
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from minify_html import minify
from html5lib.constants import DataLossWarning

from feilian.etree_tools import (
    clean_html,
    to_string,
    parse_html,
    extract_text_by_xpath,
    extract_text_by_css_selector,
)
from feilian.agents.fragments_detection import Snippet, tokenizer, run_operators
from feilian.agents.reducers import replace_with_id, append
from feilian.agents.fragments_detection import fragment_detection_graph
from feilian.chains.program_xpath_chain import (
    cot_program_xpath_s1,
    cot_program_xpath_s2,
)
from feilian.chains.program_css_selector_chain import (
    cot_program_css_selector_s1,
    cot_program_css_selector_s2,
)

warnings.filterwarnings(action="ignore", category=DataLossWarning, module=r"html5lib")

PROGRAM_FRAGMENTS = 2
PROGRAM_TYPE = "xpath"

fix_chain_state = []


class Task(TypedDict):
    field_name: str
    xpath: str


class State(TypedDict):
    snippets: Annotated[List[Snippet], replace_with_id] = []
    tasks: Annotated[List[Task], append] = []
    query: str


def fragments_detection_node(state: State) -> State:
    snippet = state["snippets"][0]
    new_state = fragment_detection_graph.invoke(
        {
            "id": snippet["id"],
            "raw_html": snippet["raw_html"],
            "ops": snippet["ops"],
            "query": state["query"],
        }
    )
    return {
        "snippets": [
            {
                "id": snippet["id"],
                "raw_html": snippet["raw_html"],
                "ops": new_state["ops"],
                "extracted": new_state["extracted"],
            }
        ],
    }


def program_node(state: State):
    tokens_before = 0
    tokens_after = 0
    htmls = []
    for snippet in state["snippets"]:
        tree = parse_html(snippet["raw_html"])
        clean_html(tree)
        tokens_before += tokenizer(minify(to_string(tree), keep_closing_tags=True))

        tree = run_operators(tree, snippet["ops"])
        html = to_string(tree)
        minified_html = minify(html, keep_closing_tags=True)
        tokens_after += tokenizer(minified_html)

        htmls.append(minified_html)

    if PROGRAM_TYPE == "xpath" and PROGRAM_FRAGMENTS == 1:
        chain = cot_program_xpath_s1
    elif PROGRAM_TYPE == "css_selector" and PROGRAM_FRAGMENTS == 1:
        chain = cot_program_css_selector_s1
    elif PROGRAM_TYPE == "css_selector" and PROGRAM_FRAGMENTS == 2:
        chain = cot_program_css_selector_s2
    elif PROGRAM_TYPE == "xpath" and PROGRAM_FRAGMENTS == 2:
        chain = cot_program_xpath_s2
    data = (
        chain.invoke(
            dict(
                htmls=htmls,
                datas=[x["extracted"] for x in state["snippets"]],
            )
        )
        or {}
    )

    tasks = []
    for field_name, xpath in data.items():
        if field_name == "_thought":
            continue

        if isinstance(xpath, list) and xpath:
            xpath = xpath[0]

        if not xpath:
            continue

        if isinstance(xpath, list):
            xpath = xpath[0]

        if isinstance(xpath, dict) and PROGRAM_TYPE == "xpath":
            xpath = " | ".join(set(xpath.values()))
        if isinstance(xpath, dict) and PROGRAM_TYPE == "css_selector":
            xpath = ", ".join(set(xpath.values()))

        tasks.append(dict(field_name=field_name, xpath=xpath))

    fix_chain_state.append(
        {
            "id": state["snippets"][0]["id"],
            "ops": state["snippets"][0]["ops"],
            "data": state["snippets"][0]["extracted"],
            "tasks": tasks,
        }
    )
    print(
        f"[Program XPath] {tokens_before} ==> {tokens_after} [{json.dumps(data, ensure_ascii=False)}]"
    )
    return dict(tasks=tasks)


def merge_node(state: State):
    return state


def unescape_and_strip(text):
    import html

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


def rank_xpath_node(state, category: str, site: str):
    """group by field name, then rank by the number of extracted data"""

    tasks = state["tasks"]
    df = pd.DataFrame(tasks)
    df = df.drop_duplicates(subset=["field_name", "xpath"])

    df["n_extracted"] = 0
    extract_fn = (
        extract_text_by_xpath
        if PROGRAM_TYPE == "xpath"
        else extract_text_by_css_selector
    )
    df["f1_score"] = 0
    for i, row in df.iterrows():
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        field_name = row["field_name"]
        xpath = row["xpath"]
        for snippet in state["snippets"]:
            extracted = extract_fn(parse_html(snippet["raw_html"]), xpath)[0]
            ground_truth = snippet["extracted"].get(field_name, [])
            tp, fp, fn = eval_array(extracted, ground_truth)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1_score = (
            2 * (precision * recall) / (precision + recall) if precision + recall else 0
        )
        df.loc[i, "n_extracted"] += len(extracted)
        df.loc[i, "f1_score"] += f1_score

    # take the first xpath
    left_df = (
        df.sort_values(["field_name", "f1_score", "n_extracted"], ascending=False)
        .groupby("field_name")
        .first()
    )
    df.drop(columns=["n_extracted"], inplace=True)
    left_df.merge(df, on=["field_name", "xpath"], how="inner")

    left_df["category"] = category
    left_df["site"] = site

    return left_df.reset_index()


def fanout_to_fragments_detection(state: State):
    return [
        Send(
            "fragments_detection",
            {
                "snippets": [snippet],
                "tasks": state["tasks"],
                "query": state["query"],
            },
        )
        for snippet in state["snippets"]
    ]


def fanout_to_program_xpath(state: State):
    if PROGRAM_FRAGMENTS == 1:
        compositions = [
            (0,),
            (1,),
            (2,),
        ]
    elif PROGRAM_FRAGMENTS == 2:
        compositions = [
            (0, 1),
            (1, 2),
            (0, 2),
        ]
    elif PROGRAM_FRAGMENTS == 3:
        compositions = [
            (0, 1, 2),
        ]

    return [
        Send(
            "program_xpath",
            {
                "snippets": [state["snippets"][i] for i in composition],
                "tasks": state["tasks"],
                "query": state["query"],
            },
        )
        for composition in compositions
    ]


def build_graph(memory=None):
    builder = StateGraph(State)

    # add nodes
    builder.add_node("fragments_detection", fragments_detection_node)
    builder.add_node("merge_node", merge_node)
    builder.add_node("program_xpath", program_node)

    # add edges
    builder.add_conditional_edges(START, fanout_to_fragments_detection)
    builder.add_edge("fragments_detection", "merge_node")
    builder.add_conditional_edges("merge_node", fanout_to_program_xpath)
    builder.add_edge("program_xpath", END)

    return builder.compile(checkpointer=memory)


def build_state(files: List[str], query: str, ids: List[str] = []) -> State:
    snippets = []
    if len(ids) == 0:
        ids = [None] * len(files)

    for file, id in zip(files, ids):
        raw_html = open(file, "r").read()
        ops = []
        id = id or hashlib.md5(raw_html.encode()).hexdigest()
        snippets.append(dict(id=id, raw_html=raw_html, tables=[], ops=ops))

    return dict(snippets=snippets, query=query)
