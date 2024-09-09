import bs4
import re
import html
from typing import List
from bs4 import BeautifulSoup
from copy import deepcopy, copy
from tokenizers import Tokenizer
from urllib.parse import unquote
from collections import defaultdict

from feilian.tools import find_most_repeated_sub_sequence_html
from feilian.html_constants import INLINE_ELEMENTS, INTERACTIVE_ELEMENTS


def _decode_url(element):
    if hasattr(element, "attrs"):
        if "href" in element.attrs:
            element.attrs["href"] = unquote(element.attrs["href"])
        if "src" in element.attrs:
            element.attrs["src"] = unquote(element.attrs["src"])


def decode_url(soup: BeautifulSoup):
    deep_first_travel(soup, _decode_url, depth=0)
    return soup


def _clean_html_with_soup(element, debug=False):
    # 移除注释等
    if isinstance(element, bs4.element.PreformattedString):
        element.extract()
        if debug:
            print(f"Remove PreformattedString: {str(element)}")
        return

    if not isinstance(element, bs4.element.Tag):
        return

    # 移除交互元素
    if hasattr(element, "name") and element.name in INTERACTIVE_ELEMENTS:
        element.extract()
        if debug:
            print(f"Remove interactive element: {str(element)}")
        return

    # 移除空白元素
    if (
        hasattr(element, "name")
        and element.name != "img"
        and element.get_text().strip() == ""
    ):
        element.extract()
        if debug:
            print(f"Remove empty element: {str(element)}")
        return

    # 移除多余属性
    if hasattr(element, "attrs"):
        element.attrs = {
            key: value
            for key, value in element.attrs.items()
            if key in ["class", "id", "title", "alt", "href", "src"]
        }

        # 移除 href="javascript:*"
        if "href" in element.attrs and element.attrs["href"].startswith("javascript:"):
            del element.attrs["href"]

        # 移除 img src
        if element.name == "img" and "src" in element.attrs:
            del element.attrs["src"]


def clean_html(soup: BeautifulSoup, debug: bool = False):
    deep_first_travel(soup, lambda x: _clean_html_with_soup(x, debug=debug))
    return soup


def extract_html_structure(soup: BeautifulSoup):
    for element in soup.find_all():
        if isinstance(element, bs4.element.Tag):
            # only keep class and id attributes
            if element.attrs:
                element.attrs = {
                    key: value for key, value in element.attrs.items() if key == "class"
                }

            if element.name in INLINE_ELEMENTS:
                element.extract()
                continue

    # remove text nodes
    for element in soup.find_all(text=True):
        element.extract()

    return soup


def deep_first_travel(element: bs4.element.Tag, callback):
    if hasattr(element, "contents"):
        for child in list(
            element.children
        ):  # !important, list() is necessary to avoid skipping children
            deep_first_travel(child, callback)

    callback(element)


def breadth_first_travel(element: bs4.element.Tag, callback, enable_interruption=False):
    queue = [element]
    while queue:
        current = queue.pop(0)

        should_interrupt = callback(current)
        if enable_interruption and should_interrupt:
            continue

        queue.extend(
            child for child in current.children if isinstance(child, bs4.element.Tag)
        )


def get_table_title(element: bs4.element.Tag):
    if element.name == "table":
        title = element.find("caption")
        if title:
            return title.get_text().strip()

    if element.previousSibling:
        return element.previousSibling.get_text().strip()

    return None


def extract_tables(element: bs4.element.Tag):
    """从当前元素中提取一层表格。"""
    tables = []

    def _extract(el: bs4.element.Tag):
        if el.name == "table":
            tables.append(
                {
                    "xpath": get_xpath(el),
                    "content": el.prettify().strip(),
                    "title": get_table_title(el),
                    "children": [],
                }
            )
            return True
        return False

    breadth_first_travel(element, _extract, enable_interruption=True)

    return tables


def extract_tables_recursive(element: bs4.element.Tag):
    """从当前元素中递归提取表格。"""
    tables = []

    def _extract(el: bs4.element.Tag):
        if el.name == "table":
            child_tables = []
            for child in el.children:
                if isinstance(child, bs4.element.Tag):
                    child_tables += extract_tables_recursive(child)

            tables.append(
                {
                    "xpath": get_xpath(el),
                    "content": el.prettify().strip(),
                    "title": get_table_title(el),
                    "children": child_tables,
                }
            )
            return True
        return False

    breadth_first_travel(element, _extract, enable_interruption=True)

    return tables


def get_tables_depth(tables: list):
    if not tables:
        return 0

    def _get_depth(table, depth):
        if not table["children"]:
            return depth

        return max(_get_depth(child, depth + 1) for child in table["children"])

    return max(_get_depth(table, 1) for table in tables)


def get_tables_width(tables: list):
    if not tables:
        return 0

    def _get_width(table):
        if not table["children"]:
            return 1

        return sum(_get_width(child) for child in table["children"])

    return sum(_get_width(table) for table in tables)


def get_tables_max_width(tables: list):
    if not tables:
        return 0

    def _get_width(table):
        if not table["children"]:
            return 1

        return max(_get_width(child) for child in table["children"])

    return max(_get_width(table) for table in tables)


def get_tables_count(tables: list):
    if not tables:
        return 0

    def _get_count(table):
        if not table["children"]:
            return 1

        return sum(_get_count(child) for child in table["children"])

    return sum(_get_count(table) for table in tables)


def _keep_unique_structure(element: bs4.element.Tag):
    if not isinstance(element, bs4.element.Tag):
        return

    # 这里相当于浅拷贝，使用 element.contents 会导致 clear() 时 contents 也被清空
    children = list(element.children)
    if not children or len(children) == 1:
        return

    if element.name == "li":
        return

    # keep the first child if they are tr with td or th
    is_tr_with_td = element.name == "tr" and children[0].name == "td"
    is_tr_with_th = element.name == "tr" and children[0].name == "th"
    if is_tr_with_td or is_tr_with_th:
        return

    # keep the first child if they are table and tr
    is_table_with_tr = element.name == "table" and children[0].name == "tr"
    is_tbody_with_tr = element.name == "tbody" and children[0].name == "tr"
    if is_table_with_tr or is_tbody_with_tr:
        element.clear()
        element.extend(children[:1])
        return

    # keep the first child if they are ul and li
    is_ul_with_li = element.name == "ul" and children[0].name == "li"
    is_ol_with_li = element.name == "ol" and children[0].name == "li"
    if is_ul_with_li or is_ol_with_li:
        element.clear()
        element.extend(children[:1])
        return

    # if every child has the same structure, keep the first one
    str_children = [str(child) for child in children]
    for i in range(len(str_children) - 1):
        if not re.match(r"^<\w+\s+class=", str_children[i]):
            continue

        current = str_children[i]
        # compare to the rest of the children
        if all(current == child for child in str_children[i + 1 :]):  # noqa
            element.clear()
            element.extend(children[: i + 1])
            return

    # test if multiple children have the same structure
    repeats = find_most_repeated_sub_sequence_html(str_children)
    if repeats:
        remove_indices = set()
        for start, end in repeats[1:]:
            remove_indices.update(range(start, end))

        keep_indices = set(range(len(children))) - remove_indices
        keep_children = [children[i] for i in keep_indices]

        element.clear()
        element.extend(keep_children)


def get_structure(html_content: str, unique=True):
    soup = BeautifulSoup(html_content, "html5lib")
    clean_html(soup)

    structure = extract_html_structure(soup)
    if unique:
        deep_first_travel(structure, _keep_unique_structure)

    return structure


def _is_same_element(e1: bs4.element.Tag, e2: bs4.element.Tag):
    if not isinstance(e1, bs4.element.Tag):
        return False

    if not isinstance(e2, bs4.element.Tag):
        return False

    if e1.name != e2.name:
        return False

    if e1.attrs or e2.attrs:
        e1_attrs = e1.attrs or {}
        e2_attrs = e2.attrs or {}

        if e1_attrs.get("class") != e2_attrs.get("class"):
            return False

    return True


def prune_by_structure(origin: BeautifulSoup, structure: BeautifulSoup):
    """根据结构修剪原始的 html 树。"""
    assert _is_same_element(
        origin, structure
    ), "The structure is not the same as the origin."

    # 检查是否是叶子节点
    if not origin.contents or not structure.contents:
        return

    # 递归地修剪
    origin_i = 0
    structure_i = 0

    while origin_i < len(origin.contents) and structure_i < len(structure.contents):
        origin_child = origin.contents[origin_i]
        structure_child = structure.contents[structure_i]

        if _is_same_element(origin_child, structure_child):
            prune_by_structure(origin_child, structure_child)

            origin_i += 1
            structure_i += 1
            continue

        if origin_child:
            origin_child.extract()
            continue

    # 删除多余的节点
    children = list(origin.children)
    origin.clear()
    origin.extend(children[:origin_i])


def prune_by_tokens(
    tokenizer: Tokenizer,
    soup: BeautifulSoup,
    max_tokens: int,
    reversed: bool = False,
):
    total_token = len(tokenizer.encode(str(soup)).ids)

    # 如果总长度小于 max_tokens，不需要修剪
    if total_token <= max_tokens:
        return

    # 如果是文字子节点，保留
    if isinstance(soup, bs4.element.NavigableString):
        return

    # 如果没有子节点，删除
    if not soup.contents:
        soup.extract()
        return

    self_node = copy(soup)
    self_node.clear()
    self_tokens = len(tokenizer.encode(str(self_node)).ids)
    required_tokens = max_tokens - self_tokens

    children = list(soup.children)
    if reversed:
        children = reversed(children)

    acc_tokens = 0
    for idx, child in enumerate(children):
        child_tokens = len(tokenizer.encode(str(child)).ids)
        if acc_tokens + child_tokens > required_tokens:
            break
        acc_tokens += child_tokens

    # 删除多余的节点
    if reversed:
        soup.contents = soup.contents[idx:]
    else:
        soup.contents = soup.contents[: idx + 1]

    # 递归修剪
    prune_by_tokens(tokenizer, child, required_tokens - acc_tokens, reversed=reversed)

    return soup


def extract_left_subset(
    soup: BeautifulSoup,
    tokenizer: Tokenizer,
    max_tokens: int = 2048,
):
    soup = deepcopy(soup)
    prune_by_tokens(tokenizer, soup, max_tokens, reversed=False)
    return soup


def get_xpath(element):
    components = []
    target = element if element.name else element.parent
    for node in (target, *target.parents)[-2::-1]:
        tag = "%s:%s" % (node.prefix, node.name) if node.prefix else node.name
        siblings = node.parent.find_all(tag, recursive=False)
        components.append(
            tag
            if len(siblings) == 1
            else "%s[%d]"
            % (
                tag,
                next(
                    index
                    for index, sibling in enumerate(siblings, 1)
                    if sibling is node
                ),
            )
        )
    return "/%s" % "/".join(components)


def get_node_depth(node: bs4.element.Tag):
    depth = 0
    while node.parent:
        node = node.parent
        depth += 1
    return depth


def get_node_contain_text(soup: BeautifulSoup, text: str):
    text = html.unescape(text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)

    d = defaultdict(int)
    for node in reversed(soup.find("body").find_all(text=True)):
        target = str(node).strip()
        if not target:
            continue

        target = html.unescape(target)
        target = html.unescape(target)
        target = re.sub(r"\s+", " ", target).strip()

        if text in target:
            d[node] = get_node_depth(node)

    if d:
        return max(d, key=d.get)
    return None


def get_common_ancestor(nodes: List[bs4.element.Tag]):
    if not nodes:
        return None

    common_ancestor = nodes[0]
    for node in nodes[1:]:
        node_parents = list(node.parents)
        for parent in [common_ancestor] + list(common_ancestor.parents):
            if parent in node_parents:
                common_ancestor = parent
                break

    return common_ancestor
