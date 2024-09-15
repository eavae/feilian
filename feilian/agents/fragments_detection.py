import tiktoken
import json
import functools
from typing_extensions import TypedDict
from typing import List, Optional, Annotated, Dict
from enum import Enum
from lxml import etree
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_openai import ChatOpenAI
from copy import deepcopy

from feilian.etree_tools import parse_html, clean_html, to_string
from feilian.etree_token_stats import extract_fragments_by_weight
from feilian.text_tools import convert_html_to_text
from feilian.prompts import EXTRACTION_PROMPT_CN, EXTRACTION_PROMPT_HISTORY
from feilian.agents.reducers import merge_operators


encoder = tiktoken.encoding_for_model("gpt-4")


class OperatorTypes(Enum):
    PRUNE = "prune"
    EXTRACT = "extract"


class Operator(TypedDict):
    xpath: str
    operator_type: Optional[OperatorTypes]
    text: Optional[str]
    data: Optional[Dict]


class Snippet(TypedDict):
    id: str
    raw_html: str
    ops: Annotated[List[Operator], merge_operators]


class FragmentDetectionState(Snippet):
    query: str


def tokenizer(text):
    if not text:
        return 0
    return len(encoder.encode(text))


def extract_fragments_node(state: FragmentDetectionState) -> FragmentDetectionState:
    ops = []
    tree = parse_html(state["raw_html"])
    clean_html(tree)

    for xpath in extract_fragments_by_weight(tree, tokenizer):
        nodes = tree.xpath(xpath)

        node_texts = []
        n: etree._Element
        for n in nodes:
            html_fragment = to_string(n)
            text = convert_html_to_text(html_fragment)
            node_texts.append(text)
            n.clear()
            n.text = ""

        text = "\n".join(node_texts)
        ops.append({"xpath": xpath, "text": text})

        print(f"[Fragment:{state['id']}] xpath: {xpath}, tokens: {tokenizer(text)}")

    return {
        "id": state["id"],
        "raw_html": state["raw_html"],
        "ops": ops,
        "query": state["query"],
    }


def _create_extraction_chain():
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.1,
        model_kwargs={
            "response_format": {
                "type": "json_object",
            },
        },
    )
    return EXTRACTION_PROMPT_CN.partial(chat_history=EXTRACTION_PROMPT_HISTORY) | llm


extraction_chain = _create_extraction_chain()


def detect_fragment_node(state: FragmentDetectionState) -> FragmentDetectionState:
    operator = state["ops"][0]
    if operator["text"] and operator["text"].strip():
        response = extraction_chain.invoke(
            {
                "context": operator["text"],
                "query": state["query"],
            }
        )
        data = json.loads(response.content)
        data = {k: v for k, v in data.items() if k != "_thought" and v}
        print(
            f"[Detection:{state['id']}] xpath: {operator['xpath']}, extracted: {data}"
        )
        return {
            "ops": [{"xpath": operator["xpath"], "data": data}],
        }
    return {"ops": [{"xpath": operator["xpath"], "data": {}}]}


def _operator_sort_fn(a: Operator, b: Operator):
    if len(a["data"]) == len(b["data"]):
        if a["xpath"] == b["xpath"]:
            return 0
        return 1 if a["xpath"] > b["xpath"] else -1
    return len(a["data"]) - len(b["data"])


def classify_fragments_node(state: FragmentDetectionState) -> FragmentDetectionState:
    ops = deepcopy(state["ops"])

    # sort ops by data fields length
    # ops.sort(key=functools.cmp_to_key(_operator_sort_fn), reverse=True)

    # classify ops by overlapping data fields
    classify = {}
    # all_keys = set()
    for op in ops:
        keys = set(op["data"].keys())
        # if keys.issubset(all_keys):
        #     classify[op["xpath"]] = OperatorTypes.PRUNE
        # else:
        #     classify[op["xpath"]] = OperatorTypes.EXTRACT
        #     all_keys.update(keys)
        if len(keys) > 0:
            classify[op["xpath"]] = OperatorTypes.EXTRACT
        else:
            classify[op["xpath"]] = OperatorTypes.PRUNE

    # fix prune ops based on tree structure
    removes = set()
    extracts = [
        x["xpath"] for x in ops if classify[x["xpath"]] == OperatorTypes.EXTRACT
    ]
    for prune_op in [op for op in ops if classify[op["xpath"]] == OperatorTypes.PRUNE]:
        if any(x.startswith(prune_op["xpath"]) for x in extracts):
            removes.add(prune_op["xpath"])

    # update operator_type
    ops = [op for op in ops if op["xpath"] not in removes]
    for op in ops:
        op["operator_type"] = classify[op["xpath"]]
        del op["data"]

    return {"ops": ops}


def fanout_to_fragment_detection(
    state: FragmentDetectionState,
) -> List[FragmentDetectionState]:
    ops = state["ops"]
    return [
        Send(
            "detect_fragment",
            {
                "id": state["id"],
                "raw_html": state["raw_html"],
                "ops": [op],
                "query": state["query"],
            },
        )
        for op in ops
    ]


def build_graph():
    builder = StateGraph(FragmentDetectionState)

    builder.add_node("extract_fragments", extract_fragments_node)
    builder.add_node("detect_fragment", detect_fragment_node)
    builder.add_node("classify_fragments", classify_fragments_node)

    builder.add_edge(START, "extract_fragments")
    builder.add_conditional_edges("extract_fragments", fanout_to_fragment_detection)
    builder.add_edge("detect_fragment", "classify_fragments")
    builder.add_edge("classify_fragments", END)

    return builder.compile()


fragment_detection_graph = build_graph()

if __name__ == "__main__":
    import pandas as pd
    import os
    import hashlib

    # from langchain.globals import set_debug

    # set_debug(True)

    category = "restaurant"
    site = "tripadvisor"
    random_state = 42
    root_dir = "data/swde"
    query = open(f"datasets/swde/questions_cn/{category}_{site}.txt", "r").read()

    df = pd.read_csv("data/swde_token_stats.csv")
    df = df[(df["category"] == category) & (df["site"] == site)]
    row = df.sample(3, random_state=random_state).iloc[0]

    html = open(os.path.join(root_dir, row["file_path"])).read()
    state = {
        "id": hashlib.md5(html.encode()).hexdigest(),
        "raw_html": html,
        "query": query,
    }
    graph = build_graph()
    result = graph.invoke(state)
    pass
