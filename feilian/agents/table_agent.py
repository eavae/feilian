import os
import hashlib
import html5lib
import json_repair
import pandas as pd
from lxml import etree
from typing import List, Annotated
from typing_extensions import TypedDict, TypeVar, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from enum import Enum
from langchain_openai import ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from minify_html import minify
from langgraph.checkpoint.sqlite import SqliteSaver

from feilian.etree_tools import (
    clean_html,
    deduplicate_to_prune,
    extraction_based_pruning,
    traverse,
    to_string,
)
from feilian.prompts import (
    EXTRACTION_PROMPT_CN,
    EXTRACTION_PROMPT_HISTORY,
    XPATH_PROGRAM_PROMPT_HISTORY_CN,
    XPATH_PROGRAM_PROMPT_CN,
    QUESTION_CONVERSION_COMP_CN,
)


T = TypeVar("T")


class ContentTypes(Enum):
    LAYOUT = "layout"  # 布局元素
    POST = "post"  # 文章内容
    TABLE = "table"  # 数据表格


class OperatorTypes(Enum):
    PRUNE = "prune"
    EXTRACT = "extract"


def replace_with_id(left: T, right: T) -> T:
    if any([not x["id"] for x in left]) or any([not x["id"] for x in right]):
        raise ValueError("id is required")

    left_ids = set([x["id"] for x in left])
    right_ids = set([x["id"] for x in right])
    token_left = left_ids - right_ids

    left_items = []
    for item in left:
        if item["id"] in token_left:
            left_items.append(item)

    return left_items + right


def append(left: List[T], right: List[T]) -> List[T]:
    return left + right


class Operator(TypedDict):
    xpath: str
    operator_type: Optional[OperatorTypes]


class Snippet(TypedDict):
    id: str
    raw_html: str
    ops: List[Operator]


class Task(TypedDict):
    field_name: str
    xpath: str


class State(TypedDict):
    snippets: Annotated[List[Snippet], replace_with_id] = []
    tasks: Annotated[List[Task], append] = []
    query: str
    xpath_query: str


def get_tree(snippet: Snippet, compact: bool = True) -> etree._Element:
    tree = html5lib.parse(
        snippet["raw_html"],
        treebuilder="lxml",
        namespaceHTMLElements=False,
    )
    if not compact:
        return tree

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
    xpaths = deduplicate_to_prune([op["xpath"] for op in extract_ops])
    extraction_based_pruning(tree, xpaths)

    return tree


def _create_table_extraction_chain():
    llm = ChatOpenAI(   #ChatAnthropic
        model=os.getenv("OPENAI_MODEL"),
        temperature=0.1,
        model_kwargs={
            "response_format": {
                "type": "json_object",
            },
        },
    )
    return EXTRACTION_PROMPT_CN.partial(chat_history=EXTRACTION_PROMPT_HISTORY) | llm


def _create_program_xpath_chain():
    llm = ChatOpenAI(  #ChatAnthropic
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
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), temperature=0.1) # ChatAnthropic
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
            }
            for xpath in xpaths
        ]

    return group_ops


def detect_tables_node(state) -> State:
    """从 snippets 获取检测表格内容，判断是否需要提取或移除"""
    snippet: Snippet = state["snippet"]
    query = state["query"]

    tree = get_tree(snippet)
    operations = snippet["ops"]

    chain = _create_table_extraction_chain()
    for node, xpath in traverse(tree):
        if node.tag != "table":
            continue

        # TODO: 统计表格的 token 数量
        #

        table = to_string(node)
        minified_table = minify(table, keep_closing_tags=True)
        if not minified_table:
            operations += [
                {
                    "id": len(operations),
                    "xpath": xpath,
                    "operator_type": OperatorTypes.PRUNE,
                }
            ]
            continue

        response = chain.invoke(
            input=dict(
                context=minified_table,
                query=query,
            )
        )
        data = json_repair.loads(response.content)
        del data["_thought"]
        data = {key: value for key, value in data.items() if value}

        if len(data) == 0:
            operations += [
                {
                    "id": len(operations),
                    "xpath": xpath,
                    "operator_type": OperatorTypes.PRUNE,
                }
            ]
        else:
            operations += [
                {
                    "id": len(operations),
                    "xpath": xpath,
                    "operator_type": OperatorTypes.EXTRACT,
                }
            ]

    # merge operations
    operations = merge_operations(operations)

    return dict(snippets=[{**snippet, "ops": operations}])


def program_xpath_node(state):
    snippet = state["snippet"]
    query = state["query"]

    tree = get_tree(snippet, compact=True)
    chain = _create_program_xpath_chain()
    html = to_string(tree)
    minified_html = minify(html, keep_closing_tags=True)
    response = chain.invoke(dict(html=minified_html, query=query))
    data = json_repair.loads(response.content)

    tasks = []
    for field_name, xpath in data.items():
        if field_name == "_thought":
            continue

        tasks.append(dict(field_name=field_name, xpath=xpath))

    return dict(tasks=tasks)


# def test_xpath_node(state) -> List[Task]:
#     """将 html 转换为文本后，使用 xpath 提取数据，然后让 LLM 进行测试"""
#     snippet: List[Snippet] = state["snippet"]
#     tasks: List[Task] = state["tasks"]

#     tree = get_tree(snippet, compact=False)
#     text = convert_html_to_text(snippet["raw_html"])

#     extracted_data = {}
#     for task in tasks:
#         ele = tree.xpath(task["xpath"])
#         extracted_data[task["field_name"]] = ele

#     return tasks


def rank_xpath_node(state):
    """group by field name, then rank by the number of extracted data"""

    tasks = state["tasks"]
    df = pd.DataFrame(tasks)
    df = df.drop_duplicates(subset=["field_name", "xpath"])

    df["n_extracted"] = 0
    for snippet in state["snippets"]:
        tree = get_tree(snippet, compact=False)
        df["n_extracted"] += df["xpath"].apply(
            lambda x: (
                1
                if len(
                    tree.xpath(
                        x, namespaces={"re": "http://exslt.org/regular-expressions"}
                    )
                )
                else 0
            )
        )

    # take the first xpath
    left_df = (
        df.sort_values(["field_name", "n_extracted"], ascending=False)
        .groupby("field_name")
        .first()
    )
    left_df.merge(df, on=["field_name", "xpath"], how="inner")
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
    builder.add_node("detect_tables", detect_tables_node)
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
