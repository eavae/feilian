import tiktoken
from typing_extensions import TypedDict
from typing import List, Optional, Annotated, Dict
from enum import Enum
from lxml import etree
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from feilian.chains.information_extraction_chain import (
    cued_information_extraction_chain,
)
from minify_html import minify
from collections import defaultdict
from copy import deepcopy

from feilian.etree_tools import (
    parse_html,
    clean_html,
    to_string,
    deduplicate_to_prune,
    extraction_based_pruning,
    gen_xpath_by_text,
)
from feilian.etree_token_stats import extract_fragments_by_weight
from feilian.html_constants import TEXT_VISUAL_PRIORITY

from feilian.agents.reducers import merge_operators


encoder = tiktoken.encoding_for_model("gpt-4")


def convert_html_to_text(html: str) -> str:
    return minify(html)


class OperatorTypes(str, Enum):
    PRUNE = "prune"
    EXTRACT = "extract"

    @staticmethod
    def from_str(value: str):
        if value == "prune":
            return OperatorTypes.PRUNE
        if value == "extract":
            return OperatorTypes.EXTRACT
        raise ValueError(f"Unknown operator type: {value}")


# fragment and extracted data
class Operator(TypedDict):
    xpath: str
    operator_type: Optional[OperatorTypes]
    text: Optional[str]
    data: Optional[Dict]


def merge_dict(a: Dict, b: Dict) -> Dict:
    return {**a, **b}


class Snippet(TypedDict):
    id: str
    raw_html: str
    ops: Annotated[List[Operator], merge_operators]
    field_operators: Annotated[
        Dict[str, Operator], merge_dict
    ]  # field name -> operators
    extracted: Optional[Dict]


class FragmentDetectionState(Snippet):
    query: str


def tokenizer(text):
    if not text:
        return 0
    return len(encoder.encode(text))


def run_operators(tree, ops: List[Operator]) -> etree._Element:
    # 2. run prune operations
    prune_ops = [o for o in ops if o["operator_type"] == OperatorTypes.PRUNE]
    xpaths = deduplicate_to_prune([op["xpath"] for op in prune_ops])
    for xpath in xpaths:
        nodes: etree._Element = tree.xpath(xpath)
        if isinstance(nodes, list):
            for node in nodes:
                node.clear()
                node.text = ""
        else:
            nodes.clear()
            nodes.text = ""

    # 3. run extract operations
    extract_ops = [o for o in ops if o["operator_type"] == OperatorTypes.EXTRACT]
    xpaths = [op["xpath"] for op in extract_ops]
    extraction_based_pruning(tree, xpaths)

    return tree


def extract_fragments_node(state: FragmentDetectionState) -> FragmentDetectionState:
    ops = []
    tree = parse_html(state["raw_html"])
    clean_html(tree, deep=True)

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

    # add /html to ops
    html_text = convert_html_to_text(to_string(tree))
    ops.append({"xpath": "/html", "text": html_text})
    print(f"[Fragment:{state['id']}] xpath: /html, tokens: {tokenizer(html_text)}")

    return {
        "id": state["id"],
        "raw_html": state["raw_html"],
        "ops": ops,
        "query": state["query"],
    }


def detect_fragment_node(state: FragmentDetectionState) -> FragmentDetectionState:
    operator = state["ops"][0]
    if operator["text"] and operator["text"].strip():
        data = cued_information_extraction_chain.invoke(
            {
                "context": operator["text"],
                "query": state["query"],
            }
        )

        # remove empties
        for k, v in list(data.items()):
            if not v:
                del data[k]
                continue

            values = v.get("value", [])
            if isinstance(values, list):
                values = [x for x in values if x]
                data[k] = {
                    "value": values,
                    "hint_text": v.get("hint_text", ""),
                }

            if not values:
                del data[k]

        print(
            f"[Detection:{state['id']}] xpath: {operator['xpath']}, extracted: {data}"
        )
        return {
            "ops": [{"xpath": operator["xpath"], "data": data}],
        }
    return {"ops": [{"xpath": operator["xpath"], "data": {}}]}


def generate_operators_for_field(
    field_name: str, operators: List[Operator]
) -> List[Operator]:
    field_related_ops = [
        op for op in operators if field_name in op["data"] and op["data"][field_name]
    ]

    if not field_related_ops:
        return []

    ops = []
    for op in operators:
        field_data = op["data"]
        is_target = field_name in field_data and field_data[field_name]
        if is_target:
            field_value = field_data[field_name]
            ops.append(
                {
                    "xpath": op["xpath"],
                    "operator_type": OperatorTypes.EXTRACT,
                    "data": field_value,
                }
            )
        else:
            ops.append(
                {
                    "xpath": op["xpath"],
                    "operator_type": OperatorTypes.PRUNE,
                }
            )

    # remove unused prune operators
    for i in reversed(range(len(ops))):
        if ops[i]["operator_type"] == OperatorTypes.PRUNE:
            del ops[i]
        else:
            break

    return ops


def group_to_field_operators_node(
    state: FragmentDetectionState,
) -> FragmentDetectionState:
    all_field_names = set(
        [
            k
            for op in state["ops"]
            for k in op["data"].keys()
            if op["data"][k].get("value")
        ]
    )

    field_operators = defaultdict(list)
    for field_name in all_field_names:
        operators = generate_operators_for_field(field_name, state["ops"])

        # if one field has multiple extract operators, ranking one
        if (
            len([x for x in operators if x["operator_type"] == OperatorTypes.EXTRACT])
            > 1
        ):
            operators = ranking_operators(operators, state["raw_html"])
        field_operators[field_name] = operators

    return dict(field_operators=field_operators)


def ranking_operators(ops: List[Operator], raw_html: str) -> List[Operator]:
    scores = []
    groups_of_operators = []
    for i in range(len(ops)):
        op = ops[i]
        if op["operator_type"] == OperatorTypes.PRUNE:
            continue

        tree = parse_html(raw_html)
        clean_html(tree)
        target_ops = deepcopy(ops[:i])
        for x in target_ops:
            x["operator_type"] = OperatorTypes.PRUNE

        operators = target_ops + [op]
        run_operators(tree, operators)
        groups_of_operators.append(operators)
        results = [gen_xpath_by_text(tree, x) for x in op["data"]["value"]]
        results = [x for x in results if x]

        # skip if no results
        if not results:
            scores.append(0)
            continue

        # calculate score of each xpath
        element_scores = []
        for result in results:
            tag = result["xpath"].split("/")[-2].split("[")[0]
            priority = (
                TEXT_VISUAL_PRIORITY.index(tag) if tag in TEXT_VISUAL_PRIORITY else 1
            )
            element_scores += [
                (len(TEXT_VISUAL_PRIORITY) - priority) / len(TEXT_VISUAL_PRIORITY)
            ]
        element_score = sum(element_scores) / (len(element_scores) + 1e-8)

        # calculate text length
        text_scores = []
        for result in results:
            additional_length = len(result["in_text"]) - len(result["target_text"])
            text_scores.append(1 - additional_length / len(result["in_text"]))
        text_score = sum(text_scores) / (len(text_scores) + 1e-8)

        # score
        score = element_score * 0.5 + text_score * 0.5
        scores.append(score)

    print(f"[Ranking] scores: {scores}")
    best_idx = scores.index(max(scores))
    print(f"[Ranking] best idx: {best_idx}")
    return groups_of_operators[best_idx]


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
    builder.add_node("group_to_field_operators", group_to_field_operators_node)

    builder.add_edge(START, "extract_fragments")
    builder.add_conditional_edges("extract_fragments", fanout_to_fragment_detection)
    builder.add_edge("detect_fragment", "group_to_field_operators")
    builder.add_edge("group_to_field_operators", END)

    return builder.compile()


fragment_detection_graph = build_graph()
