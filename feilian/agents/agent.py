import os
import hashlib
import json
import pandas as pd
from lxml import etree
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from enum import Enum
from langchain_openai import ChatOpenAI
from minify_html import minify
from langgraph.checkpoint.sqlite import SqliteSaver

from feilian.etree_tools import (
    clean_html,
    deduplicate_to_prune,
    extraction_based_pruning,
    to_string,
    parse_html,
)
from feilian.agents.fragments_detection import Snippet, Operator, OperatorTypes
from feilian.agents.reducers import replace_with_id, append
from feilian.prompts import (
    EXTRACTION_PROMPT_CN,
    EXTRACTION_PROMPT_HISTORY,
    XPATH_PROGRAM_PROMPT_HISTORY_CN,
    XPATH_PROGRAM_PROMPT_CN,
    QUESTION_CONVERSION_COMP_CN,
)


class ContentTypes(Enum):
    LAYOUT = "layout"  # 布局元素
    POST = "post"  # 文章内容
    TABLE = "table"  # 数据表格


class Task(TypedDict):
    field_name: str
    xpath: str


class State(TypedDict):
    snippets: Annotated[List[Snippet], replace_with_id] = []
    tasks: Annotated[List[Task], append] = []
    query: str
    xpath_query: str


def get_tree(snippet: Snippet) -> etree._Element:
    tree = parse_html(snippet["raw_html"])

    # 1. clean html
    clean_html(tree)

    # 2. run prune operations
    prune_ops = [o for o in snippet["ops"] if o["operator_type"] == OperatorTypes.PRUNE]
    xpaths = deduplicate_to_prune([op["xpath"] for op in prune_ops])
    for xpath in xpaths:
        ele: etree._Element = tree.xpath(xpath)
        if isinstance(ele, list):
            for x in ele:
                x.clear()
                x.text = "..."
        else:
            ele.clear()
            ele.text = "..."

    # 3. run extract operations
    extract_ops = [
        o for o in snippet["ops"] if o["operator_type"] == OperatorTypes.EXTRACT
    ]
    xpaths = [op["xpath"] for op in extract_ops]
    extraction_based_pruning(tree, xpaths)

    return tree


def _create_table_extraction_chain():
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


def _create_program_xpath_chain():
    llm = ChatOpenAI(
        model="deepseek-coder",
        temperature=0.1,
        model_kwargs={
            "response_format": {
                "type": "json_object",
            },
        },
    )
    return (
        XPATH_PROGRAM_PROMPT_CN.partial(chat_history=XPATH_PROGRAM_PROMPT_HISTORY_CN)
        | llm
    )


def _create_question_conversion_chain():
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.1)
    return QUESTION_CONVERSION_COMP_CN | llm


def query_conversion_node(state: State) -> State:
    if state["xpath_query"] is not None:
        return dict(xpath_query=state["xpath_query"])

    chain = _create_question_conversion_chain()
    response = chain.invoke(dict(query=state["query"]))
    return dict(xpath_query=response.content)


def merge_operations(operations: List[Operator]) -> List[Operator]:
    ops = []

    class_group = {OperatorTypes.PRUNE: [], OperatorTypes.EXTRACT: []}
    for o in operations:
        class_group[o["operator_type"]].append(o)

    for k, group_ops in class_group.items():
        xpaths = deduplicate_to_prune([o["xpath"] for o in group_ops])
        ops += [
            {
                "id": len(ops),
                "xpath": xpath,
                "operator_type": k,
                "content_type": ContentTypes.LAYOUT,
            }
            for xpath in xpaths
        ]

    return group_ops


def program_xpath_node(state):
    snippet = state["snippet"]
    query = state["query"]

    tree = get_tree(snippet)
    chain = _create_program_xpath_chain()
    html = to_string(tree)
    minified_html = minify(html)
    response = chain.invoke(dict(html=minified_html, query=query))
    data = json.loads(response.content)

    tasks = []
    for field_name, xpath in data.items():
        if field_name == "_thought":
            continue

        tasks.append(dict(field_name=field_name, xpath=xpath))

    return dict(tasks=tasks)


def rank_xpath_node(state):
    """group by field name, then rank by the number of extracted data"""

    tasks = state["tasks"]
    df = pd.DataFrame(tasks)
    df = df.drop_duplicates(subset=["field_name", "xpath"])

    df["n_extracted"] = 0
    for snippet in state["snippets"]:
        tree = parse_html(snippet["raw_html"])
        df["n_extracted"] += df["xpath"].apply(lambda x: 1 if len(tree.xpath(x)) else 0)

    # take the first xpath
    left_df = (
        df.sort_values(["field_name", "n_extracted"], ascending=False)
        .groupby("field_name")
        .first()
    )
    left_df = left_df.merge(df, on=["field_name", "xpath"], how="inner")
    return left_df


def fanout_to_table_detection(state: State):
    return [
        Send("detect_tables", {"snippet": snippet, "query": state["query"]})
        for snippet in state["snippets"]
    ]


def fanout_to_program_xpath(state: State):
    return [
        Send("program_xpath", {"snippet": snippet, "query": state["xpath_query"]})
        for snippet in state["snippets"]
    ]


def build_graph(memory=None):
    builder = StateGraph(State)

    # add nodes
    builder.add_node("query_conversion", query_conversion_node)
    builder.add_node("fragments_detection", fragments_detection_node)
    builder.add_node("program_xpath", program_xpath_node)
    # builder.add_node("rank_xpath", rank_xpath_node)
    # builder.add_node("test_xpath", test_xpath_node)

    # add edges
    builder.add_edge(START, "query_conversion")
    builder.add_conditional_edges("query_conversion", fanout_to_table_detection)
    builder.add_conditional_edges("detect_tables", fanout_to_program_xpath)
    builder.add_edge("program_xpath", END)

    return builder.compile(checkpointer=memory)


def build_state(files: List[str], query: str) -> State:
    snippets = []
    for file in files:
        raw_html = open(file, "r").read()
        ops = []
        snippets.append(
            dict(
                id=hashlib.md5(raw_html.encode()).hexdigest(),
                raw_html=raw_html,
                tables=[],
                ops=ops,
            )
        )

    return dict(snippets=snippets, query=query, xpath_query=None)


if __name__ == "__main__":
    import sqlite3

    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    category = "university"
    site = "embark"
    random_state = 42
    root_dir = "data/swde"
    query = open(f"datasets/swde/questions_cn/{category}_{site}.txt", "r").read()

    df = pd.read_csv("swde_token_stats.csv")
    df = df[(df["category"] == category) & (df["site"] == site)]
    df = df.sample(5, random_state=random_state)

    files = [os.path.join(root_dir, x) for x in df["file_path"]]
    graph = build_graph()
    state = build_state(files, query)

    # config = {"configurable": {"thread_id": "2"}}
    state = graph.invoke(state)
    df = rank_xpath_node(state)
    df.to_csv("ranked_xpaths.csv", index=False)
    # state = graph.get_state(config)
    # for event in graph.stream(state, config):
    #     print(event)
    #     pass
