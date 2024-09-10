import numpy as np
from typing import List, Callable
from lxml import etree
from collections import defaultdict
from feilian.html_constants import INLINE_ELEMENTS, CONTAINER_ELEMENTS

MIN_TEXT_TOKEN = 256
MIN_HTML_TOKEN = 512
MAX_TEXT_TOKEN = 1024
MAX_HTML_TOKEN = 2048


def logistic(x):
    return 1 / (1 + np.exp(-5 * (x / 2 - 1)))


class Node:
    xpath: str
    token: int
    ele: etree._Element
    children: List["Node"]
    parent: "Node" = None
    depth: int
    weight: int

    def __init__(self, xpath: str, token: int, ele: etree._Element, depth):
        self.xpath = xpath
        self.token = token
        self.ele = ele
        self.children = []
        self.depth = depth

    @property
    def max_depth(self):
        if not self.children:
            return self.depth
        return max(child.max_depth for child in self.children)

    @property
    def max_token(self):
        if not self.children:
            return self.token
        return max(child.max_token for child in self.children)

    @property
    def leafs(self):
        if not self.children:
            return 1
        return sum(child.leafs for child in self.children)

    @property
    def width(self):
        return len(self.children)

    @property
    def max_width(self):
        if not self.children:
            return 1

        self_width = len(self.children) or 1
        return max(self_width, max(child.max_width for child in self.children))

    @property
    def most_weighted_node(self):
        if not self.children:
            return self
        weighted_children = [child.most_weighted_node for child in self.children]
        weighted_children.append(self)
        return max(weighted_children, key=lambda x: x.weight)

    def reweighing(
        self, max_depth: int, total_token: int, max_width: int, only_text=True
    ):
        depth_weight = self.depth / max_depth
        token_weight = np.tanh(self.token / total_token)
        width_weight = self.width / max_width

        element_weight = 0.6
        tag = self.ele.tag.lower()
        is_container_element = tag in CONTAINER_ELEMENTS
        is_inline_element = tag in INLINE_ELEMENTS
        has_class = len(self.ele.attrib.get("class", "")) > 0
        if is_inline_element and not has_class:
            element_weight = 0.4
        elif is_inline_element and has_class:
            element_weight = 0.7
        elif is_container_element and not has_class:
            element_weight = 0.9
        elif is_container_element and has_class:
            element_weight = 1
        elif tag == "div" and has_class:
            element_weight = 0.8

        weight = depth_weight + token_weight + width_weight + element_weight
        min_token = only_text and MIN_TEXT_TOKEN or MIN_HTML_TOKEN
        max_token = only_text and MAX_TEXT_TOKEN or MAX_HTML_TOKEN
        if self.token < min_token:
            weight = 0
        elif self.token > max_token:
            weight = 0
        self.weight = weight

        for child in self.children:
            child.reweighing(max_depth, total_token, max_width)

    def add_child(self, child: "Node"):
        child.parent = self
        self.children.append(child)

    def add_children(self, children: List["Node"]):
        for child in children:
            self.add_child(child)


def _build_token_tree(
    root: etree._Element,
    xpath: str,
    depth: int,
    tokenizer: Callable,
    only_text=True,
) -> Node:
    # children
    tag_counts = defaultdict(int)
    for ele in root.iterchildren():
        tag_counts[ele.tag] += 1

    children: List[Node] = []
    tag_order = defaultdict(int)
    ele: etree._Element
    for ele in root.iterchildren():
        new_xpath = f"{xpath}/{ele.tag}"
        if tag_counts[ele.tag] > 1:
            new_xpath = f"{xpath}/{ele.tag}[{tag_order[ele.tag] + 1}]"
        tag_order[ele.tag] += 1
        children.append(
            _build_token_tree(
                ele,
                new_xpath,
                depth + 1,
                tokenizer,
                only_text,
            )
        )

    # count token, not accurate, but enough for comparison
    text = root is not None and root.text or ""
    text = text.strip()
    text_token = tokenizer(text)
    if only_text:
        token = sum(child.token for child in children) + text_token
    else:
        attr_str = " ".join(f"{k}='{v}'" for k, v in root.items())
        element_str = f"<{root.tag} {attr_str}></{root.tag}>"
        element_token = tokenizer(element_str)
        token = sum(child.token for child in children) + text_token + element_token

    node = Node(xpath=xpath, token=token, ele=root, depth=depth)
    node.add_children(children)

    return node


def build_token_tree(
    tree: etree._ElementTree | etree._Element, tokenizer: Callable, only_text=True
) -> Node:

    if isinstance(tree, etree._ElementTree):
        root = tree.getroot()
        if root is None:
            raise ValueError("root is None")
        if root.tag != "html":
            raise ValueError("root tag is not html")
        xpath = "/html"
    else:
        root = tree
        xpath = f"/{tree.tag}"

    return _build_token_tree(root, xpath, 1, tokenizer, only_text)


def find_node(node: Node, token_below: int) -> Node:
    if not node.children:
        return node

    max_token = 0
    max_node = None
    for child in node.children:
        if child.token > max_token:
            max_token = child.token
            max_node = child

    if max_token <= token_below:
        return max_node

    return find_node(max_node, token_below)


def remove_node(node: Node):
    parent = node.parent
    node.parent.children.remove(node)
    node.parent = None
    while parent:
        parent.token -= node.token
        parent = parent.parent


def remove_node_until(
    tree: Node, token_below: int = 1024, until: int = 4096
) -> tuple[int, List[int]]:
    times = 0
    remove_tokens = []

    while tree.token > until:
        node = find_node(tree, token_below)
        remove_node(node)

        times += 1
        remove_tokens.append(node.token)
    remove_tokens.append(tree.token)

    return times, remove_tokens


def extract_fragments_by_weight(
    tree: etree._ElementTree | etree._Element,
    tokenizer: Callable,
    only_text=True,
    until: int = 468,
):
    xpath = []

    token_tree = build_token_tree(tree, tokenizer, only_text)
    while True:
        if token_tree.token < until:
            break

        token_tree.reweighing(
            token_tree.max_depth,
            token_tree.token,
            token_tree.max_width,
            only_text,
        )
        node = token_tree.most_weighted_node

        if not node:
            break
        if node.depth <= 1:
            break

        remove_node(node)
        xpath.append(node.xpath)

    return xpath
