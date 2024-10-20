import re
import html5lib
import html
from lxml import etree
from tokenizers import Tokenizer
from copy import deepcopy
from urllib.parse import unquote
from typing import List, Optional
from collections import defaultdict
from lxml.cssselect import CSSSelector
from functools import partial

from feilian.html_constants import INTERACTIVE_ELEMENTS, INVISIBLE_ELEMENTS

from feilian.text_tools import convert_html_to_text, normalize_text


# A regex matching the "invalid XML character range"
ILLEGAL_XML_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def strip_illegal_xml_characters(s, default, base=10):
    # Compare the "invalid XML character range" numerically
    n = int(s, base)
    if (
        n in (0xB, 0xC, 0xFFFE, 0xFFFF)
        or 0x0 <= n <= 0x8
        or 0xE <= n <= 0x1F
        or 0xD800 <= n <= 0xDFFF
    ):
        return ""
    return default


def remove_control_characters(html_str: str):
    """
    Strip invalid XML characters that `lxml` cannot parse.
    """

    # See: https://github.com/html5lib/html5lib-python/issues/96
    #
    # The XML 1.0 spec defines the valid character range as:
    # Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    #
    # We can instead match the invalid characters by inverting that range into:
    # InvalidChar ::= #xb | #xc | #xFFFE | #xFFFF | [#x0-#x8] | [#xe-#x1F] | [#xD800-#xDFFF]
    #
    # Sources:
    # https://www.w3.org/TR/REC-xml/#charsets,
    # https://lsimons.wordpress.com/2011/03/17/stripping-illegal-characters-out-of-xml-in-python/

    # We encode all non-ascii characters to XML char-refs, so for example "💖" becomes: "&#x1F496;"
    # Otherwise we'd remove emojis by mistake on narrow-unicode builds of Python
    html_str = html_str.encode("ascii", "xmlcharrefreplace").decode("utf-8")
    html_str = re.sub(
        r"&#(\d+);?",
        lambda c: strip_illegal_xml_characters(c.group(1), c.group(0)),
        html_str,
    )
    html_str = re.sub(
        r"&#[xX]([0-9a-fA-F]+);?",
        lambda c: strip_illegal_xml_characters(c.group(1), c.group(0), base=16),
        html_str,
    )
    html_str = ILLEGAL_XML_CHARS_RE.sub("", html_str)
    return html_str


def parse_html(html_str: str):
    html_str = remove_control_characters(html_str)
    return html5lib.parse(html_str, treebuilder="lxml", namespaceHTMLElements=False)


def post_order_traversal(tree: etree._Element, func):
    for ele in tree.iterchildren():
        post_order_traversal(ele, func)

    func(tree)


def _traverse(root: etree._Element, xpath: str):
    # children
    tag_counts = defaultdict(int)
    for ele in root.iterchildren():
        tag_counts[ele.tag] += 1

    tag_order = defaultdict(int)
    ele: etree._Element
    for ele in root.iterchildren():
        new_xpath = f"{xpath}/{ele.tag}"
        if tag_counts[ele.tag] > 1:
            new_xpath = f"{xpath}/{ele.tag}[{tag_order[ele.tag] + 1}]"
        tag_order[ele.tag] += 1

        yield from _traverse(ele, new_xpath)
        yield (ele, new_xpath)

    yield (root, xpath)


def traverse(tree: etree._Element | etree._ElementTree):
    if isinstance(tree, etree._ElementTree):
        root = tree.getroot()
        if root is None:
            raise ValueError("root is None")
        if root.tag != "html":
            raise ValueError("root tag is not html")

        return _traverse(root, "/html")

    return _traverse(tree, f"/{tree.tag}")


def _pre_order_traversal(tree: Optional[etree._Element], xpath, func):
    if tree is None:
        return

    # pre-order
    func(tree, xpath)

    # children
    tag_counts = defaultdict(int)
    for ele in tree.iterchildren():
        tag_counts[ele.tag] += 1

    tag_order = defaultdict(int)
    for ele in tree.iterchildren():
        if tag_counts[ele.tag] > 1:
            _pre_order_traversal(
                ele, f"{xpath}/{ele.tag}[{tag_order[ele.tag] + 1}]", func
            )
        else:
            _pre_order_traversal(ele, f"{xpath}/{ele.tag}", func)
        tag_order[ele.tag] += 1


def pre_order_traversal(tree: etree._Element | etree._ElementTree, func):
    if isinstance(tree, etree._ElementTree):
        root: etree._Element = tree.getroot()
        _pre_order_traversal(root, f"/{root.tag}", func)
    else:
        _pre_order_traversal(tree, f"/{tree.tag}", func)


def breadth_first_travel(element: etree._Element, callback, enable_interruption=False):
    queue = [element]
    while queue:
        current = queue.pop(0)
        if current is None:
            continue

        should_interrupt = callback(current)
        if enable_interruption and should_interrupt:
            continue

        queue.extend(current.getchildren())


def _remove(element: etree._Element):
    p = element.getparent()
    if p is not None:
        p.remove(element)


def _is_empty(ele: etree._Element):
    is_no_text = ele.text is None or not ele.text.strip()
    is_no_tail = ele.tail is None or not ele.tail.strip()
    is_no_children = len(ele.getchildren()) == 0
    return is_no_text and is_no_tail and is_no_children


def _clean_html(ele: etree._Element, deep=False):
    # 移除非元素的节点
    if not isinstance(ele, etree._Element):
        _remove(ele)
        return

    # 移除注释 ele.tag.__name__ == "Comment"
    if not isinstance(ele.tag, str) and ele.tag.__name__ == "Comment":
        _remove(ele)
        return

    # 移除交互元素
    if ele.tag in INTERACTIVE_ELEMENTS:
        _remove(ele)
        return

    # 移除 head
    if ele.tag in INVISIBLE_ELEMENTS:
        _remove(ele)
        return

    if deep:
        # 移除图片
        if ele.tag == "img":
            _remove(ele)
            return

        # 移除空白节点
        if _is_empty(ele):
            _remove(ele)
            return

    # 移除 display:none
    if "style" in ele.attrib and re.search(r"display\s*:\s*none", ele.attrib["style"]):
        ele.clear()
        ele.text = ""
        return

    # 移除多余属性
    if ele.attrib:
        if deep:
            for key in list(ele.attrib.keys()):
                del ele.attrib[key]
        else:
            for key in list(ele.attrib.keys()):
                if key not in ["class", "id"]:
                    del ele.attrib[key]

            # 移除 href="javascript:*"
            if "href" in ele.attrib and ele.attrib["href"].startswith("javascript:"):
                del ele.attrib["href"]

            # 移除 img src
            if ele.tag == "img" and "src" in ele.attrib:
                del ele.attrib["src"]


def clean_html(ele: etree._Element | etree._ElementTree, deep=False):
    if isinstance(ele, etree._ElementTree):
        post_order_traversal(ele.getroot(), partial(_clean_html, deep=deep))
    else:
        post_order_traversal(ele, partial(_clean_html, deep=deep))
    return ele


def remove_children(ele: etree._Element):
    for child in ele.getchildren():
        ele.remove(child)
    return ele


def to_string(ele: etree._Element, pretty_print=False):
    html_str = etree.tostring(ele, encoding="utf-8").decode("utf-8")
    if pretty_print:
        from bs4 import BeautifulSoup

        return BeautifulSoup(html_str, "html.parser").prettify()
    return html_str


def prune_by_tokens(
    tokenizer: Tokenizer,
    ele: etree._Element,
    max_tokens: int,
    reversed: bool = False,
):
    if ele is None:
        return

    # 如果总长度小于 max_tokens，不需要修剪
    total_token = len(tokenizer.encode(to_string(ele)).ids)
    if total_token <= max_tokens:
        return

    # check children
    children = ele.getchildren()
    remove_children(ele)
    self_tokens = len(tokenizer.encode(to_string(ele)).ids)
    required_tokens = max_tokens - self_tokens
    if reversed:
        children = reversed(children)

    # no children
    if len(children) == 0:
        return

    acc_tokens = 0
    for idx, child in enumerate(children):
        child_tokens = len(tokenizer.encode(to_string(child)).ids)
        if acc_tokens + child_tokens > required_tokens:
            break
        acc_tokens += child_tokens

    # 保留需要的子节点
    if reversed:
        ele.extend(reversed(children[: idx + 1]))
    else:
        ele.extend(children[: idx + 1])

    # 递归修剪
    prune_by_tokens(tokenizer, child, required_tokens - acc_tokens, reversed=reversed)

    return ele


def parent_xpath(xpath: str):
    return "/".join(xpath.split("/")[:-1])


def get_text_content(ele: etree._Element):
    arr = []
    for t in ele.itertext():
        t = t.strip()
        if t:
            arr.append(t)
    return " ".join(arr)


def replace_with_text(ele: etree._Element):
    text = get_text_content(ele)
    remove_children(ele)
    ele.text = text


def prune_to_text(ele: etree._Element):
    """
    修剪为文本节点
    """
    if len(ele) == 0 and ele.text:
        ele.text = ele.text.strip()
        return

    # 仅保留 td
    if ele.tag == "tr":
        for child in ele.getchildren():
            if child.tag == "td":
                replace_with_text(child)
    # TODO: prune table
    elif ele.tag == "table":
        return ele
    elif ele.tag in {"ul", "ol"}:
        for child in ele.getchildren():
            if child.tag == "li":
                replace_with_text(child)
    else:
        replace_with_text(ele)
    return ele


def deduplicate_to_prune(xpaths: List[str]):
    """
    去重 xpath，保留最上层的节点
    """
    xpaths = sorted(xpaths)
    remove_indexes = set()
    for i in range(len(xpaths)):
        xpath = xpaths[i]

        for j in range(i + 1, len(xpaths)):
            if xpaths[j].startswith(xpath):
                remove_indexes.add(j)

    return [xpaths[i] for i in range(len(xpaths)) if i not in remove_indexes]


def prune_by_xpath(
    ele: etree._Element,
    xpath: str,
    includes: List[str] = [],
):
    """
    根据 xpath 进行修剪
    保留 includes 周围的节点
    return bool: 是否应继续遍历
    """
    is_in_path = any([x.startswith(xpath) for x in includes])
    is_contained = any([xpath.startswith(x) for x in includes])
    if not is_in_path and not is_contained:
        include_parent = any([x.startswith(parent_xpath(xpath)) for x in includes])
        if include_parent:
            for child in ele.getchildren():
                ele.remove(child)
            if ele.text:
                ele.text = "..."
            if ele.tail:
                ele.tail = "..."
            return False

    return True


def extract_left_subtree(
    tokenizer: Tokenizer,
    element: etree._Element,
    max_tokens: int = 2048,
):
    element = deepcopy(element)
    prune_by_tokens(tokenizer, element, max_tokens, reversed=False)
    return element


def _decode_url(element: etree._Element):
    if element.attrib:
        if "href" in element.attrib:
            element.attrib["href"] = unquote(element.attrib["href"])

        if "src" in element.attrib:
            element.attrib["src"] = unquote(element.attrib["src"])


def decode_url(element: etree._Element):
    post_order_traversal(element, _decode_url)
    return element


def apply_trim_rules(root: etree._Element, rules: List[str]):
    for rule in rules:
        for ele in root.xpath(
            rule, namespaces={"re": "http://exslt.org/regular-expressions"}
        ):
            ele.getparent().remove(ele)
    return root


def extraction_based_pruning(
    tree: etree._Element | etree._ElementTree, includes: List[dict]
):
    """
    prune tree nodes except includes
    """
    pre_order_traversal(
        tree,
        lambda ele, xpath: prune_by_xpath(ele, xpath, includes=includes),
    )


def remove_by_xpath(tree: etree._Element, xpath: str):
    for ele in tree.xpath(
        xpath, namespaces={"re": "http://exslt.org/regular-expressions"}
    ):
        ele.getparent().remove(ele)


def extract_text_by_xpath(tree: etree._Element | etree._ElementTree, xpath: str):
    if not isinstance(xpath, str):
        return []

    results = []
    try:
        for ele in tree.xpath(xpath):
            if ele is None:
                continue
            if isinstance(ele, str):
                results.append(ele)
            else:
                results.append(convert_html_to_text(to_string(ele)))
    except Exception:
        print(f"Invalid xpath: {xpath}")
        return [], True

    results: List[str] = [normalize_text(x) for x in results]
    results = [x.strip() for x in results if x.strip()]

    return results, False


def extract_text_by_css_selector(tree: etree._Element, css_selector: str):
    try:
        selector = CSSSelector(css_selector)
    except Exception:
        print(f"Invalid css selector: {css_selector}")
        return [], True
    elements = selector(tree)

    results: List[str] = [
        html.unescape(convert_html_to_text(to_string(ele))) for ele in elements
    ]
    results = [x.strip() for x in results if x.strip()]
    results = [re.sub(r"  +", " ", x) for x in results]

    return results, False


def get_predicates(ele: etree._Element, with_id=True, with_class=True):
    part_str = ""
    if ele.attrib:
        parts = []
        if with_id and "id" in ele.attrib:
            parts.append(f"@id=\"{ele.attrib['id']}\"")
        elif with_class and "class" in ele.attrib:
            parts.append(f"@class=\"{ele.attrib['class']}\"")

        if parts:
            part_str = "[" + " and ".join(parts) + "]"
    return part_str


def get_xpath(ele, short=True, with_id=True, with_class=True):
    xpath = ""
    while ele is not None:
        parent = ele.getparent()
        if parent is None:
            xpath = f"/{ele.tag}{xpath}"
            break

        part_str = get_predicates(ele, with_id=with_id, with_class=with_class)

        idx = 0
        cur_idx = 0
        for e in parent:
            if part_str and e.tag == ele.tag and get_predicates(e) == part_str:
                idx += 1
            elif not part_str and e.tag == ele.tag:
                idx += 1

            if e == ele:
                cur_idx = idx

        if idx == 1:
            xpath = f"/{ele.tag}{part_str}{xpath}"
        else:
            xpath = f"/{ele.tag}{part_str}[{cur_idx}]{xpath}"

        if short and ele.attrib and "id" in ele.attrib:
            xpath = "/" + xpath
            break

        ele = ele.getparent()

    return xpath


def itertext(ele):
    idx = 1
    tag = ele.tag
    if not isinstance(tag, str) and tag is not None:
        return
    t = ele.text
    if t:
        yield (ele, t, idx)
        idx += 1

    for e in ele:
        yield from itertext(e)
        t = e.tail
        if t:
            yield (ele, t, idx)
            idx += 1


def gen_xpath_by_text(
    tree: etree._Element | etree._ElementTree,
    target_text: str,
    text_suffix: bool = False,
    short: bool = True,
    with_id: bool = True,
    with_class: bool = True,
):
    target_text = normalize_text(target_text)

    root = tree
    if isinstance(tree, etree._ElementTree):
        root = tree.getroot()

    results = []
    for ele, text, idx in itertext(root):
        processed_text = normalize_text(text)
        if not processed_text:
            continue

        if target_text in processed_text or processed_text in target_text:
            results.append(
                {
                    "element": ele,
                    "text_idx": idx,
                    "target_text": target_text,
                    "in_text": str(text),
                }
            )

    if not results:
        return []

    scores = [abs(len(x["in_text"]) - len(x["target_text"])) for x in results]
    min_score = min(scores)
    indices = [i for i, x in enumerate(scores) if x == min_score]
    results = [results[i] for i in indices]

    xpaths = []
    for result in results:
        xpath = get_xpath(
            result["element"], short=short, with_id=with_id, with_class=with_class
        )
        if text_suffix:
            if result["text_idx"] > 1:
                xpath = f"{xpath}/text()[{result['text_idx']}]"
            else:
                xpath = f"{xpath}/text()"
        xpaths.append(xpath)

    return xpaths
