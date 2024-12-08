import hashlib
import warnings
import json
import os
from typing import List, Annotated, Dict, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from minify_html import minify
from html5lib.constants import DataLossWarning

from feilian.etree_tools import (
    clean_html,
    to_string,
    parse_html,
    gen_xpath_by_text,
    extraction_based_pruning,
    extract_text_by_xpath,
)
from feilian.agents.reducers import replace_with_id
from feilian.chains.information_extraction_chain import (
    cued_information_extraction_chain,
)
from feilian.chains.program_xpath_chat import (
    program_xpath_chat_chain,
    format_snippets,
    format_feedbacks,
)

warnings.filterwarnings(action="ignore", category=DataLossWarning, module=r"html5lib")


class Snippet(TypedDict):
    id: str
    raw_html: str
    data: Optional[Dict]


def merge_dict(x: Dict, y: Dict) -> Dict:
    return {**x, **y}


def unique_merge(x: List, y: List) -> List:
    return list(dict.fromkeys(x + y))


class State(TypedDict):
    snippets: Annotated[List[Snippet], replace_with_id] = []
    fields: Annotated[List[str], unique_merge] = []
    xpaths: Annotated[Dict[str, List[str]], merge_dict] = {}
    query: str


def information_extraction_node(state: State) -> State:
    snippet = state["snippets"][0]

    # skip if data already exists
    if snippet.get("data"):
        return dict(snippets=[snippet])

    html = snippet["raw_html"]
    tree = parse_html(html)
    clean_html(tree, deep=True)
    cleaned_html = minify(to_string(tree), keep_closing_tags=True)

    data = cued_information_extraction_chain.invoke(
        dict(query=state["query"], context=cleaned_html)
    )

    # filter empty values
    for field_name, value_object in list(data.items()):
        value = value_object.get("value", [])
        if isinstance(value, str):
            value = [value]
        values = [v.strip() for v in value_object["value"] if v.strip()]
        if values:
            value_object["value"] = values
        else:
            del data[field_name]

        # if cue_text is array
        cue_text = value_object.get("cue_text", "")
        if isinstance(cue_text, list):
            cue_texts = [t for t in cue_text if t.strip()]
            if cue_texts:
                value_object["cue_text"] = cue_texts[0]

    return {
        "snippets": [
            {
                "id": snippet["id"],
                "raw_html": snippet["raw_html"],
                "data": data,
            }
        ],
    }


infer_xpath_kwargs = {
    "text_suffix": True,
    "short": True,
    "with_id": True,
    "with_class": True,
}

extract_xpath_kwargs = {
    "text_suffix": False,
    "short": False,
    "with_id": False,
    "with_class": False,
}


def get_feedbacks(snippets, field_name, field_xpath, trees):
    feedbacks = []
    for snippet in snippets:
        tree = trees[snippet["id"]]
        field_object = snippet["data"].get(field_name, None)
        if not field_object:
            continue

        values = field_object["value"]
        if not values:
            continue

        # extract
        texts, invalid = extract_text_by_xpath(tree, field_xpath)
        missing = set(values) - set(texts)
        surplus = set(texts) - set(values)
        if missing or surplus:
            messages = []
            if missing:
                messages.append(f"Missing: {json.dumps(list(missing))}")
            if surplus:
                messages.append(f"Surplus: {json.dumps(list(surplus))}")
            if invalid:
                messages.append(f"Invalid XPath: {field_xpath}")

            feedbacks.append(
                {
                    "id": snippet["id"],
                    "extracted": texts,
                    "ground_truth": values,
                    "message": ", ".join(messages),
                }
            )
    return feedbacks


def select_best_xpath(tried_xpaths):
    # filter out invalid xpaths
    valid_xpaths = [
        (xpath, feedbacks)
        for xpath, feedbacks in tried_xpaths
        if not any("Invalid" in f["message"] for f in feedbacks)
    ]
    if len(valid_xpaths) == 0:
        return tried_xpaths[0][0]
    if len(valid_xpaths) == 1:
        return valid_xpaths[0][0]

    # filter out empty extracted
    tried_xpaths = valid_xpaths
    valid_xpaths = [
        (xpath, feedbacks)
        for xpath, feedbacks in valid_xpaths
        if all(f["extracted"] for f in feedbacks)
    ]
    if len(valid_xpaths) == 0:
        return tried_xpaths[0][0]

    # sort by extracted count (from least to most)
    valid_xpaths = sorted(
        valid_xpaths,
        key=lambda x: sum(len(f["extracted"]) for f in x[1]),
    )
    return valid_xpaths[0][0]


def program_xpath_node(state: State):
    field_name = state["fields"][0]

    # skip if xpath already exists
    if state["xpaths"].get(field_name):
        return state

    trees = {}
    for snippet in state["snippets"]:
        raw_html = minify(snippet["raw_html"], keep_closing_tags=True)
        tree = parse_html(raw_html)
        trees[snippet["id"]] = tree

    # get html snippets
    html_snippets = []
    for snippet in state["snippets"]:
        raw_html = minify(snippet["raw_html"], keep_closing_tags=True)
        tree = parse_html(raw_html)

        field_object = snippet["data"].get(field_name, {})
        if not field_object:
            continue

        values = field_object.get("value", [])
        if not values:
            continue

        # get xpath for values
        target_xpath = [
            x_path
            for v in values
            for x_path in gen_xpath_by_text(tree, v, **infer_xpath_kwargs)
        ]
        extract_xpath = [
            x_path
            for v in values
            for x_path in gen_xpath_by_text(tree, v, **extract_xpath_kwargs)
        ]
        if not target_xpath:
            continue

        cue_text = field_object.get("cue_text", None)
        if os.environ.get("ABLATION_EXPERIMENT", None) == "WITHOUT_CUE":
            cue_text = None
        if cue_text:
            cue_xpath = gen_xpath_by_text(tree, cue_text, **infer_xpath_kwargs)
            if cue_xpath:
                extract_xpath.extend(
                    gen_xpath_by_text(tree, cue_text, **extract_xpath_kwargs)
                )
        else:
            cue_xpath = []

        # prune html tree by extract xpath
        extraction_based_pruning(tree, extract_xpath)
        clean_html(tree)
        cleaned_html = to_string(tree)

        html_snippets.append(
            {
                "id": snippet["id"],
                "target_text": values,
                "target_xpath": target_xpath,
                "cue_text": cue_text,
                "cue_xpath": cue_xpath,
                "html": cleaned_html,
            }
        )

    if not html_snippets:
        return dict(xpaths={field_name: None})

    base_input = format_snippets(html_snippets)
    session_id = "_".join([snippet["id"] for snippet in state["snippets"]]) + f"_{field_name}"
    config = {"configurable": {"session_id": session_id}}
    field_xpath = program_xpath_chat_chain.invoke({"input": base_input}, config=config)
    feedbacks = get_feedbacks(state["snippets"], field_name, field_xpath, trees)

    tried_xpaths = [(field_xpath, feedbacks)]
    max_iter = 3
    while feedbacks and max_iter > 0:
        feedback_input = format_feedbacks(feedbacks)
        new_xpath = program_xpath_chat_chain.invoke(
            {"input": feedback_input}, config=config
        )
        if new_xpath in tried_xpaths:
            # select the best xpath
            field_xpath = select_best_xpath(tried_xpaths)
            break

        field_xpath = new_xpath
        feedbacks = get_feedbacks(state["snippets"], field_name, field_xpath, trees)
        tried_xpaths.append((field_xpath, feedbacks))
        max_iter -= 1

    if max_iter == 0:
        field_xpath = select_best_xpath(tried_xpaths)

    return {
        "xpaths": {field_name: field_xpath},
    }


def merge_node(state: State):
    return state


def fanout_to_information_extraction(state: State):
    return [
        Send(
            "information_extraction",
            {
                "snippets": [snippet],
                "query": state["query"],
            },
        )
        for snippet in state["snippets"]
    ]


def fanout_to_program_xpath(state: State):
    field_names = dict.fromkeys(
        [key for snippet in state["snippets"] for key in snippet["data"]]
    )

    return [
        Send(
            "program_xpath",
            {
                "snippets": state["snippets"],
                "fields": [field_name],
                "query": state["query"],
                "xpaths": state["xpaths"],
            },
        )
        for field_name in field_names
    ]


def build_graph(memory=None):
    if memory is None:
        from langgraph.checkpoint.memory import MemorySaver

        memory = MemorySaver()
    builder = StateGraph(State)

    # add nodes
    builder.add_node("information_extraction", information_extraction_node)
    builder.add_node("merge_node", merge_node)
    builder.add_node("program_xpath", program_xpath_node)

    # add edges
    builder.add_conditional_edges(START, fanout_to_information_extraction)
    builder.add_edge("information_extraction", "merge_node")
    builder.add_conditional_edges("merge_node", fanout_to_program_xpath)
    builder.add_edge("program_xpath", END)

    return builder.compile(checkpointer=memory)


def build_state(files: List[str], query: str, ids: List[str] = []) -> State:
    snippets = []
    if len(ids) == 0:
        ids = [None] * len(files)

    for file, id in zip(files, ids):
        raw_html = open(file, "r").read()
        id = id or hashlib.md5(raw_html.encode()).hexdigest()
        snippets.append(dict(id=id, raw_html=raw_html))

    return dict(snippets=snippets, query=query, fields=[], xpaths={})
