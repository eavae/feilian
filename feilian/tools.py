import re
from typing import List
from hashlib import md5


def find_most_repeated_sub_sequence_html(arr: List[str]):
    """
    Find the most repeated continuous subsequence in the given array.
    For example:
        [1, 2, 3, 4, 1, 2, 3, 4, 5] -> [(0, 4), (4, 8)]
        [0, 1, 2, 3, 1, 2, 1, 2, 1, 2] -> [(4, 6), (6, 8), (8, 10)]
    """
    # md5 the array
    _arr = [md5(x.encode()).hexdigest() for x in arr]

    max_repeated = 0
    most_repeats = []
    for i in range(len(_arr)):
        for j in range(i + 1, len(_arr) + 1):
            subsequence = _arr[i:j]

            # 检查这个子序列是否是一个合法的、有class属性的 html 标签
            # 用来防止子序列为 <div> <div> 这种情况
            if not any(re.match(r"^<\w+\s+class=", s) for s in arr[i:j]):
                continue

            # 回溯，看这个子序列重复了几次
            count = 0
            n = len(subsequence)
            repeats = [(i, j)]
            for k in range(i - n, 0, -n):
                if _arr[k : k + n] == subsequence:  # noqa
                    count += 1
                    repeats.insert(0, (k, k + n))
                else:
                    break

            if count > max_repeated:
                max_repeated = count
                most_repeats = repeats

    if len(most_repeats) < 2:
        return None

    return most_repeats


def uri_params(params, spider):
    return {**params, "spider_name": spider.name}
