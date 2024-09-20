import os
import hashlib
import json
import json_repair
import pandas as pd
import warnings
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from minify_html import minify
from html5lib.constants import DataLossWarning

from feilian.etree_tools import (
    clean_html,
    to_string,
    parse_html,
)
from feilian.agents.fragments_detection import Snippet, tokenizer, run_operators
from feilian.agents.reducers import replace_with_id, append
from feilian.prompts import (
    XPATH_PROGRAM_PROMPT_HISTORY_CN,
    XPATH_PROGRAM_PROMPT_CN,
    QUESTION_CONVERSION_COMP_CN,
    COT_PROGRAM_XPATH_PROMPT_CN,
)
from feilian.agents.fragments_detection import fragment_detection_graph

warnings.filterwarnings(action="ignore", category=DataLossWarning, module=r"html5lib")


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
            }
        ],
    }


def _create_program_xpath_chain():
    def parser(response):
        return json_repair.repair_json(
            response.content,
            return_objects=True,
        )

    llm = ChatOpenAI(
        model="deepseek-coder",
        temperature=0,
        model_kwargs={
            "response_format": {
                "type": "json_object",
            },
        },
    )
    return (
        XPATH_PROGRAM_PROMPT_CN.partial(chat_history=XPATH_PROGRAM_PROMPT_HISTORY_CN)
        | llm
        | parser
    )


program_xpath = _create_program_xpath_chain()


def _create_cot_program_xpath_chain():
    def parser(response):
        json_str = response.content.split("最终结论:")[-1].strip()
        json_str = json_str.split("```")[0].strip()
        return json_repair.repair_json(json_str, return_objects=True)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
    )
    return PromptTemplate.from_template(COT_PROGRAM_XPATH_PROMPT_CN) | llm | parser


cot_program_xpath = _create_cot_program_xpath_chain()


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

    while True:
        try:
            data = cot_program_xpath.invoke(
                dict(
                    query=query,
                    html0=htmls[0],
                    html1=htmls[1],
                    html2=htmls[2],
                )
            )
            if data:
                break
        except Exception as e:
            print(f"Error: {e}")
            continue

    tasks = []
    for field_name, xpath in data.items():
        if field_name == "_thought":
            continue

        if isinstance(xpath, list):
            xpath = xpath[0]

        if not xpath:
            continue

        if isinstance(xpath, list):
            xpath = xpath[0]

        if isinstance(xpath, dict):
            xpath = " | ".join(set(xpath.values()))

        tasks.append(dict(field_name=field_name, xpath=xpath))

    print(
        f"[Program XPath] {tokens_before} ==> {tokens_after} [{json.dumps(data, ensure_ascii=False)}]"
    )
    return dict(tasks=tasks)


def robust_xpath(tree, xpath):
    try:
        return tree.xpath(
            xpath, namespaces={"re": "http://exslt.org/regular-expressions"}
        )
    except Exception as e:
        print(f"Error: {e}")
        return []


def rank_xpath_node(state, category: str, site: str):
    """group by field name, then rank by the number of extracted data"""

    tasks = state["tasks"]
    df = pd.DataFrame(tasks)
    df = df.drop_duplicates(subset=["field_name", "xpath"])

    df["n_extracted"] = 0
    for snippet in state["snippets"]:
        tree = parse_html(snippet["raw_html"])
        df["n_extracted"] += df["xpath"].apply(
            lambda x: (1 if len(robust_xpath(tree, x)) else 0)
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


def build_graph(memory=None):
    builder = StateGraph(State)

    # add nodes
    builder.add_node("query_conversion", query_conversion_node)
    builder.add_node("fragments_detection", fragments_detection_node)
    builder.add_node("program_xpath", program_xpath_node)

    # add edges
    builder.add_edge(START, "query_conversion")
    builder.add_conditional_edges("query_conversion", fanout_to_fragments_detection)
    builder.add_edge("fragments_detection", "program_xpath")
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
