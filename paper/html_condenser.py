# 伪代码
from collections import defaultdict


# input:
#     root: html root node
#     target_texts: list of target texts to keep
#     d: distance between two texts
# output:
#     root: condensed html root node
target_texts = []
distances = defaultdict(int)
keep_elements = defaultdict(list)
for ele, text in itertext(root):
    for target_text in target_texts:
        distance = d(text, target_text)
        if distance < distances[text]:
            distances[text] = distance
            keep_elements[text] = [get_xpath(ele)]
        elif distance == distances[text]:
            keep_elements[text].append(get_xpath(ele))

includes = concat(keep_elements.values())
for xpath, ele in iter_elements(root):
    is_parent = any([x.startswith(xpath) for x in includes])
    is_inside = any([xpath.startswith(x) for x in includes])
    if not is_in_path and not is_contained:
        if any([x.startswith(parent_xpath(xpath)) for x in includes]):
            remove_children(ele)
            add_text(ele, "...")
