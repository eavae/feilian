import tiktoken
import pandas as pd
import os
import bs4
import json
import html
from minify_html import minify
from tqdm import tqdm

from feilian.soup_tools import clean_html

SWDE_DATA_ROOT = "data/swde"

tqdm.pandas()

encoder = tiktoken.encoding_for_model("gpt-4")


def swde__stats_token_row(row):
    html_file_path = os.path.join(SWDE_DATA_ROOT, row["file_path"])
    html_content, cleaned_html = read_and_clean_html(html_file_path)

    row["raw_tokens"] = len(encoder.encode(html_content))
    row["cleaned_tokens"] = len(encoder.encode(cleaned_html))
    return row


def swde__stats_token(dataset_file_path: str):
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=4)

    df = pd.read_parquet(dataset_file_path)

    # sample 256 per group, group by category, site
    df = df.groupby(["category", "site"]).apply(
        lambda x: x.sample(min(len(x), 256), replace=False)
    )

    # remove index
    df.reset_index(drop=True, inplace=True)

    df = df.parallel_apply(swde__stats_token_row, axis=1)
    df.to_csv("swde_token_stats.csv")


def swde__token_dist(
    dataset_file_path: str,
    group_cols: list = ["category", "site"],
    target_col: str = "cleaned_tokens",
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    df = pd.read_csv(dataset_file_path, index_col=0)
    pairs = df[group_cols].groupby(group_cols).nunique().index
    n_cols = 8
    n_rows = len(pairs) // n_cols + 1

    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=[f"{pair}" for pair in pairs]
    )

    for i, pair in enumerate(pairs):
        row, col = i // n_cols + 1, i % n_cols + 1
        pair_df = df[(df[group_cols] == pair).all(axis=1)]
        fig.add_trace(
            go.Histogram(x=pair_df[target_col], name=f"{pair}", histnorm="percent"),
            row=row,
            col=col,
        )

    fig.update_layout(height=2048)
    fig.show()
    pass


def read_and_clean_html(file_path: str):
    with open(file_path, "r") as f:
        html_content = f.read()
    soup = bs4.BeautifulSoup(html_content, "html5lib")
    cleaned_soup = clean_html(soup)
    cleaned_html = minify(cleaned_soup.prettify().strip())

    return html_content, cleaned_html


def swde__test_semantic_pruning_row(row):
    html_file_path = os.path.join(SWDE_DATA_ROOT, row["file_path"])
    _, cleaned_html = read_and_clean_html(html_file_path)

    values = set()
    attributes = json.loads(row["attributes"])
    for xs in attributes.values():
        for x in xs:
            x = x.strip()
            values.add(x)

    matches = []
    for value in values:
        if value in cleaned_html:
            matches.append(True)
            continue

        if html.unescape(value) in cleaned_html:
            matches.append(True)
            continue

        matches.append(False)

    row["semantic_pruning"] = all(matches)
    return row


def swde__test_semantic_pruning(dataset_file_path: str):
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=1)

    df = pd.read_csv(dataset_file_path)
    df = df.parallel_apply(swde__test_semantic_pruning_row, axis=1)
    # df = df.progress_apply(swde__test_semantic_pruning_row, axis=1)

    failed_case = df[df["semantic_pruning"] == False]  # noqa
    print(f"Failed cases: {len(failed_case)}")
    failed_case.to_csv("semantic_pruning_failed_cases.csv")


def swde__token_stats(df: pd.DataFrame):
    pass


if __name__ == "__main__":
    # swde__test_file(
    #     "data/swde/sourceCode/sourceCode/job/job-careerbuilder(2000)/0947.htm"
    # )

    swde__token_dist("swde_token_stats.csv")
    pass
