from typing_extensions import TypeVar, TYPE_CHECKING
from typing import List

if TYPE_CHECKING:
    from feilian.agents.fragments_detection import Operator

T = TypeVar("T")


def replace_with_id(left: T, right: T) -> T:
    if any([not x["id"] for x in left]) or any([not x["id"] for x in right]):
        raise ValueError("id is required")

    left_ids = set([x["id"] for x in left])
    right_ids = set([x["id"] for x in right])
    taken_left = left_ids - right_ids

    left_items = []
    for item in left:
        if item["id"] in taken_left:
            left_items.append(item)

    return left_items + right


def append(left: List[T], right: List[T]) -> List[T]:
    return left + right


def merge_operators(
    left: List["Operator"], right: List["Operator"]
) -> List["Operator"]:
    if not right:
        return left
    if not left:
        return right

    # 在分类时，用右边的完全替换
    if all([x.get("operator_type", None) is not None for x in right]):
        return right

    # 默认时，用右边的替换
    right_dict = {x["xpath"]: x for x in right}
    left_dict = {x["xpath"]: x for x in left}
    return [right_dict[x["xpath"]] if x["xpath"] in right_dict else x for x in left] + [
        x for x in right if x["xpath"] not in left_dict
    ]
