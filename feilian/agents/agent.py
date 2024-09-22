import os
import hashlib
import json
import pandas as pd
import warnings
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_openai import ChatOpenAI
from minify_html import minify
from html5lib.constants import DataLossWarning

from feilian.etree_tools import clean_html, to_string, parse_html, extract_text_by_xpath
from feilian.agents.fragments_detection import Snippet, tokenizer, run_operators
from feilian.agents.reducers import replace_with_id, append
from feilian.prompts import QUESTION_CONVERSION_COMP_CN
from feilian.agents.fragments_detection import fragment_detection_graph
from feilian.chains.program_xpath_chain import cot_program_xpath_s1

warnings.filterwarnings(action="ignore", category=DataLossWarning, module=r"html5lib")

PROGRAM_XPATH_FRAGMENTS = 1

fix_chain_state = []


class Task(TypedDict):
    field_name: str
    xpath: str


class State(TypedDict):
    snippets: Annotated[List[Snippet], replace_with_id] = []
    tasks: Annotated[List[Task], append] = []
    query: str
    xpath_query: str


def _create_question_conversion_chain():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0)
    return QUESTION_CONVERSION_COMP_CN | llm


def query_conversion_node(state: State) -> State:
    if state["xpath_query"] is not None:
        return dict(xpath_query=state["xpath_query"])

    chain = _create_question_conversion_chain()
    response = chain.invoke(dict(query=state["query"]))
    return dict(xpath_query=response.content)


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


def program_xpath_node(state: State):
    query = state["xpath_query"]

    tokens_before = 0
    tokens_after = 0
    htmls = []
    for snippet in state["snippets"]:
        tree = parse_html(snippet["raw_html"])
        clean_html(tree)
        tokens_before += tokenizer(minify(to_string(tree)))

        tree = run_operators(tree, snippet["ops"])
        html = to_string(tree)
        minified_html = minify(html)
        tokens_after += tokenizer(minified_html)

        htmls.append(minified_html)

    data = (
        cot_program_xpath_s1.invoke(
            dict(
                query=query,
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

        if isinstance(xpath, dict):
            xpath = " | ".join(set(xpath.values()))

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


def rank_xpath_node(state, category: str, site: str):
    """group by field name, then rank by the number of extracted data"""

    tasks = state["tasks"]
    df = pd.DataFrame(tasks)
    df = df.drop_duplicates(subset=["field_name", "xpath"])

    df["n_extracted"] = 0
    for snippet in state["snippets"]:
        tree = parse_html(snippet["raw_html"])
        df["n_extracted"] += df["xpath"].apply(
            lambda x: (1 if len(extract_text_by_xpath(tree, x)) else 0)
        )

    # take the first xpath
    left_df = (
        df.sort_values(["field_name", "n_extracted"], ascending=False)
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
                "xpath_query": state["xpath_query"],
            },
        )
        for snippet in state["snippets"]
    ]


def fanout_to_program_xpath(state: State):
    if PROGRAM_XPATH_FRAGMENTS == 1:
        compositions = [
            (0,),
            (1,),
            (2,),
        ]
    elif PROGRAM_XPATH_FRAGMENTS == 2:
        compositions = [
            (0, 1),
            (1, 2),
            (0, 2),
        ]
    elif PROGRAM_XPATH_FRAGMENTS == 3:
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
                "xpath_query": state["xpath_query"],
            },
        )
        for composition in compositions
    ]


def build_graph(memory=None):
    builder = StateGraph(State)

    # add nodes
    builder.add_node("query_conversion", query_conversion_node)
    builder.add_node("fragments_detection", fragments_detection_node)
    builder.add_node("program_xpath", program_xpath_node)
    builder.add_node("merge_node", merge_node)

    # add edges
    builder.add_edge(START, "query_conversion")
    builder.add_conditional_edges("query_conversion", fanout_to_fragments_detection)
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

    return dict(snippets=snippets, query=query, xpath_query=None)
