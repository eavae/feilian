if not root_node:
    return []

left_stack = [root_node]
right_stack = []

while left_stack:
    node = left_stack.pop()
    right_stack.append(node)
    left_stack += list(node.iterchildren())

while right_stack:
    node = right_stack.pop()
    if is_invisiable_or_no_text(node):
        node.getparent().remove(node)
    else:
        node.remove_attributes()
