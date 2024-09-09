from typing import List, Callable
from lxml import etree
from collections import defaultdict


class Node:
    xpath: str
    token: int
    ele: etree._Element
    children: List["Node"]
    parent: "Node" = None

    def __init__(self, xpath: str, token: int, ele: etree._Element):
        self.xpath = xpath
        self.token = token
        self.ele = ele
        self.children = []

    def add_child(self, child: "Node"):
        child.parent = self
        self.children.append(child)

    def add_children(self, children: List["Node"]):
        for child in children:
            self.add_child(child)


def _build_token_tree(root: etree._Element, xpath: str, tokenizer: Callable) -> Node:
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
        children.append(_build_token_tree(ele, new_xpath, tokenizer))

    # count token, not accurate, but enough for comparison
    text = root is not None and root.text or ""
    text_token = tokenizer(text)
    attr_str = " ".join(f"{k}='{v}'" for k, v in root.items())
    element_str = f"<{root.tag} {attr_str}></{root.tag}>"
    element_token = tokenizer(element_str)
    token = sum(child.token for child in children) + text_token + element_token

    node = Node(xpath=xpath, token=token, ele=root)
    node.add_children(children)

    return node


def build_token_tree(
    tree: etree._ElementTree | etree._Element, tokenizer: Callable
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

    return _build_token_tree(root, xpath, tokenizer)


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
